"""Tests for evals.metaclaw.mirix_skill_evolver."""
from unittest.mock import AsyncMock

import pytest

from evals.metaclaw.format_adapter import RoundResult
from evals.metaclaw.mirix_skill_evolver import MirixSkillEvolver


@pytest.mark.asyncio
async def test_evolve_serializes_rounds_to_messages_and_returns_metaclaw_skills():
    mirix = AsyncMock()
    mirix.evolve.return_value = {
        "created": [
            {"name": "iso8601", "description": "d", "instructions": "i",
             "entry_type": "guide", "version": "0.1.0"},
        ],
        "edited": [],
        "deleted": [],
    }

    evolver = MirixSkillEvolver(mirix_client=mirix)

    rounds = [
        RoundResult("r1", "file_check", "Q1", "A1", 0.0, "fail", "FB1"),
        RoundResult("r2", "multi_choice", "Q2", "\\bbox{A}", 1.0, "pass", "FB2"),
    ]

    out = await evolver.evolve(rounds, current_skills={})

    # Serialization: 2 rounds → 2 messages
    args, _ = mirix.evolve.call_args
    sent_messages = args[0]
    assert len(sent_messages) == 2
    assert "r1" in sent_messages[0]
    assert "r2" in sent_messages[1]

    # Output: metaclaw-shaped skills
    assert out == [
        {"name": "iso8601", "description": "d", "content": "i", "category": "guide"}
    ]


def test_should_evolve_always_true_to_let_driver_decide():
    evolver = MirixSkillEvolver(mirix_client=None)
    assert evolver.should_evolve([]) is True
    assert evolver.should_evolve([1, 2, 3], threshold=0.99) is True
