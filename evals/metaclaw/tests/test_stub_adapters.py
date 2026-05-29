"""Unit tests for the stub SkillManager / SkillEvolver adapters.

These tests assert the duck-type surfaces api_server.py reads are covered.
They do NOT spin up the proxy — they directly call every public method on
``StubSkillsAdapter`` and ``StubEvolverAdapter`` and assert no exception
escapes.  This is what protects the ``METACLAW_SKILLS_PROVIDER=stub`` end-to-end
smoke run from regressing if api_server.py grows new duck-type calls.
"""

from __future__ import annotations

import asyncio

import pytest

from evals.metaclaw.mirix_adapters._stub import (
    StubEvolverAdapter,
    StubSkillsAdapter,
)


# --------------------------------------------------------------------------- #
# StubSkillsAdapter                                                            #
# --------------------------------------------------------------------------- #


def test_stub_skills_initial_attributes():
    a = StubSkillsAdapter()
    assert a.generation == 0
    assert isinstance(a.skills, dict)
    assert a.skills["general_skills"] == []
    assert a.skills["task_specific_skills"] == {}
    assert a.skills["common_mistakes"] == []


def test_stub_skills_retrieve_returns_empty_list():
    a = StubSkillsAdapter()
    assert a.retrieve("any task", top_k=6) == []
    assert a.retrieve("", top_k=1) == []


def test_stub_skills_retrieve_relevant_returns_empty_list():
    a = StubSkillsAdapter()
    assert a.retrieve_relevant("any task", top_k=6) == []


def test_stub_skills_format_for_conversation_returns_empty_string():
    a = StubSkillsAdapter()
    assert a.format_for_conversation([]) == ""
    assert a.format_for_conversation([{"name": "ignored"}]) == ""


def test_stub_skills_add_skill_returns_false():
    a = StubSkillsAdapter()
    assert a.add_skill({"name": "x"}) is False
    assert a.add_skill({}) is False


def test_stub_skills_add_skills_returns_zero():
    a = StubSkillsAdapter()
    assert a.add_skills([{"name": "x"}], category="general") == 0
    assert a.add_skills([], category="task") == 0


def test_stub_skills_get_skill_count_returns_zero_dict():
    a = StubSkillsAdapter()
    counts = a.get_skill_count()
    assert isinstance(counts, dict)
    assert counts["general_skills"] == 0
    assert counts["task_specific_skills"] == 0
    assert counts["common_mistakes"] == 0


def test_stub_skills_reload_and_save_are_noops():
    a = StubSkillsAdapter()
    assert a.reload() is None
    assert a.save() is None
    assert a.save(path="/tmp/anything.json") is None


# --------------------------------------------------------------------------- #
# StubEvolverAdapter                                                           #
# --------------------------------------------------------------------------- #


def test_stub_evolver_initial_attributes():
    e = StubEvolverAdapter()
    assert e.update_history == []
    assert e.history_path is None


def test_stub_evolver_should_evolve_returns_true():
    e = StubEvolverAdapter()
    assert e.should_evolve([], threshold=0.4) is True
    assert e.should_evolve([1, 2, 3], threshold=0.9) is True


def test_stub_evolver_evolve_returns_empty_list():
    e = StubEvolverAdapter()
    result = asyncio.run(e.evolve([], {}))
    assert result == []
    # And with realistic-shaped inputs:
    samples = [object(), object()]
    skills = {
        "general_skills": [{"name": "g1"}],
        "task_specific_skills": {"chat": [{"name": "t1"}]},
        "common_mistakes": [],
    }
    result = asyncio.run(e.evolve(samples, skills))
    assert result == []


# --------------------------------------------------------------------------- #
# Cross-check against the real api_server duck-type surface                    #
# --------------------------------------------------------------------------- #


def test_stub_skills_covers_api_server_surface():
    """Every attribute api_server.py touches on skill_manager exists on the stub."""
    a = StubSkillsAdapter()
    for attr in (
        "skills",
        "generation",
        "retrieve",
        "retrieve_relevant",
        "format_for_conversation",
        "add_skill",
        "add_skills",
        "get_skill_count",
        "reload",
        "save",
    ):
        assert hasattr(a, attr), f"StubSkillsAdapter missing attribute: {attr}"


def test_stub_evolver_covers_api_server_surface():
    """Every attribute api_server.py touches on skill_evolver exists on the stub."""
    e = StubEvolverAdapter()
    for attr in ("evolve", "should_evolve", "update_history", "history_path"):
        assert hasattr(e, attr), f"StubEvolverAdapter missing attribute: {attr}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
