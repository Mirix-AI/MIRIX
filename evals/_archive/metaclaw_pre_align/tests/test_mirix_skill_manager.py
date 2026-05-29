"""Tests for evals.metaclaw.mirix_skill_manager."""
from unittest.mock import AsyncMock, MagicMock

from evals.metaclaw.mirix_skill_manager import MirixSkillManager


def test_retrieve_returns_metaclaw_shaped_skills():
    mirix = MagicMock()
    mirix.search_skills = AsyncMock(return_value=[
        {"name": "iso8601", "description": "d", "instructions": "i",
         "entry_type": "guide", "version": "0.1.0"},
    ])

    mgr = MirixSkillManager(mirix_client=mirix)

    out = mgr.retrieve("datetime format please", top_k=3)

    mirix.search_skills.assert_called_once()
    call_kwargs = mirix.search_skills.call_args.kwargs
    assert call_kwargs.get("limit") == 3
    assert out == [
        {"name": "iso8601", "description": "d", "content": "i", "category": "guide"}
    ]


def test_retrieve_with_no_skills_returns_empty():
    mirix = MagicMock()
    mirix.search_skills = AsyncMock(return_value=[])

    mgr = MirixSkillManager(mirix_client=mirix)
    assert mgr.retrieve("anything") == []
