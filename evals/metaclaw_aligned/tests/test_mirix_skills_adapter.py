"""Tests for MirixSkillsAdapter (PRD D5 + Testing Decisions).

httpx mock fixtures — no real MIRIX server needed. All tests complete
in under 10s total per PRD.
"""
from __future__ import annotations

import httpx
import pytest

from evals.metaclaw_aligned.mirix_skills_adapter import (
    MirixSkillsAdapter,
    _legacy_procedural_to_paper,
    _mirix_skill_to_paper,
)


# -- shape converters ------------------------------------------------------

def test_mirix_skill_to_paper_shape():
    src = {
        "id": "sk_123",
        "name": "iso8601-fix",
        "description": "fix iso8601",
        "instructions": "step 1\nstep 2",
        "entry_type": "workflow",
        "version": 3,
    }
    out = _mirix_skill_to_paper(src)
    assert out == {
        "name": "iso8601-fix",
        "description": "fix iso8601",
        "content": "step 1\nstep 2",
        "category": "workflow",
    }


def test_mirix_skill_to_paper_empty_entry_type_defaults_general():
    out = _mirix_skill_to_paper({"name": "x", "description": "y", "instructions": "z"})
    assert out["category"] == "general"


def test_legacy_procedural_steps_as_list_joined():
    row = {"summary": "do iso", "steps": ["one", "two", "three"], "entry_type": "workflow"}
    out = _legacy_procedural_to_paper(row)
    assert out["content"] == "one\ntwo\nthree"
    assert out["name"] == "workflow"


def test_legacy_procedural_steps_as_string_passes_through():
    row = {"summary": "x", "steps": "plain text", "entry_type": "guide"}
    out = _legacy_procedural_to_paper(row)
    assert out["content"] == "plain text"


def test_legacy_procedural_entry_type_defaults_procedure():
    row = {"summary": "x", "steps": "y"}
    out = _legacy_procedural_to_paper(row)
    assert out["name"] == "procedure"
    assert out["category"] == "procedure"


# -- adapter constructor / config ------------------------------------------

def test_invalid_variant_rejected():
    with pytest.raises(ValueError, match="variant must be"):
        MirixSkillsAdapter(variant="bogus", base_url="http://x", user_id="u")


def test_skill_evolve_variant_targets_8531():
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://127.0.0.1:8531", user_id="eval")
    try:
        assert a.base_url == "http://127.0.0.1:8531"
        assert a.variant == "skill-evolve"
    finally:
        a.close()


def test_legacy_variant_targets_8532():
    a = MirixSkillsAdapter(variant="legacy", base_url="http://127.0.0.1:8532", user_id="eval")
    try:
        assert a.base_url == "http://127.0.0.1:8532"
        assert a.variant == "legacy"
    finally:
        a.close()


def test_initial_skills_dict_shape_matches_paper():
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://x", user_id="u")
    try:
        # paper SkillManager exposes self.skills as a dict with these keys
        assert "general_skills" in a.skills
        assert "task_specific_skills" in a.skills
        assert "common_mistakes" in a.skills
        assert a.generation == 0
    finally:
        a.close()


# -- retrieve ---------------------------------------------------------------

def _make_adapter_with_mock(variant: str, transport: httpx.MockTransport) -> MirixSkillsAdapter:
    a = MirixSkillsAdapter(
        variant=variant,
        base_url="http://test",
        user_id="eval-test",
    )
    a._http.close()
    a._http = httpx.Client(
        base_url=a.base_url,
        transport=transport,
        headers={"X-Client-Id": a.client_id},
    )
    return a


def _user_resolve_response() -> httpx.Response:
    """Stock response for /users/create_or_get — adapter resolves human name
    to server.user.id on first call."""
    return httpx.Response(200, json={
        "id": "user-resolved-1234",
        "name": "eval-test",
        "organization_id": "org-test",
        "status": "active",
    })


def test_skills_property_reflects_backend_state():
    """Confound #2 (codex parity review): paper api_server.py:2134 reads
    self.skill_manager.skills as the dedup context for evolver. We must
    return a live snapshot of the MIRIX bank bucketed into paper's
    {general_skills, task_specific_skills, common_mistakes} shape so the
    paper evolver sees what's actually stored."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            return _user_resolve_response()
        if req.url.path == "/memory/search":
            return httpx.Response(200, json={"results": [
                {"name": "g1", "description": "[paper-category=general] gd", "instructions": "gc"},
                {"name": "w1", "description": "[paper-category=workflow] wd", "instructions": "wc", "entry_type": "workflow"},
                {"name": "c1", "description": "[paper-category=common_mistakes] cd", "instructions": "cc"},
                {"name": "x1", "description": "[paper-category=communication] xd", "instructions": "xc"},
            ]})
        return httpx.Response(404)
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        s = a.skills
        # bucket by decoded paper-category
        assert len(s["general_skills"]) == 1
        assert s["general_skills"][0]["name"] == "g1"
        assert len(s["common_mistakes"]) == 1
        assert "communication" in s["task_specific_skills"]
        assert "workflow" in s["task_specific_skills"]
    finally:
        a.close()


def test_skills_property_setter_is_noop():
    """Paper SkillManager has self.skills = dict; some code may assign.
    We're backend-backed so setter must not raise."""
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://x", user_id="u")
    try:
        a.skills = {"general_skills": []}  # must not raise
    finally:
        a.close()


def test_ensure_user_id_posts_user_id_field_not_id():
    """Confound #3 (codex parity review): server schema is
    `{user_id, name}` not `{id, name}`. Passing `id` causes server to
    treat user_id as missing and generate a fresh UUID every call →
    skills don't accumulate across process restarts. This test pins
    the payload shape so the bug can't reappear."""
    captured = {}
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            captured["body"] = httpx.Response(200, content=req.content).json()
            return _user_resolve_response()
        return httpx.Response(404)
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        a._ensure_user_id()
        assert "user_id" in captured["body"], f"missing user_id field; sent: {captured['body']}"
        assert captured["body"]["user_id"] == "eval-test"
        assert captured["body"]["name"] == "eval-test"
    finally:
        a.close()


def test_skill_evolve_retrieve_returns_paper_shape():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            return _user_resolve_response()
        assert req.url.path == "/memory/search"
        assert req.url.params.get("memory_type") == "procedural"
        assert req.url.params.get("query") == "fix dates"
        assert req.url.params.get("limit") == "3"
        assert req.url.params.get("user_id") == "user-resolved-1234"
        return httpx.Response(200, json={
            "results": [
                {"name": "s1", "description": "d1", "instructions": "c1", "entry_type": "workflow"},
                {"name": "s2", "description": "d2", "instructions": "c2", "entry_type": "guide"},
            ],
        })
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        out = a.retrieve("fix dates", top_k=3)
        assert len(out) == 2
        assert out[0] == {"name": "s1", "description": "d1", "content": "c1", "category": "workflow"}
    finally:
        a.close()


def test_legacy_retrieve_uses_procedural_search_params():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            return _user_resolve_response()
        assert req.url.path == "/memory/search"
        params = dict(req.url.params)
        assert params["memory_type"] == "procedural"
        assert params["search_method"] == "bm25"
        assert params["search_field"] == "summary"
        assert params["query"] == "fix dates"
        assert params["limit"] == "4"
        assert params["user_id"] == "user-resolved-1234"
        return httpx.Response(200, json={
            "results": [
                {"summary": "use iso8601", "steps": ["a", "b"], "entry_type": "workflow"},
            ],
        })
    a = _make_adapter_with_mock("legacy", httpx.MockTransport(handler))
    try:
        out = a.retrieve("fix dates", top_k=4)
        assert len(out) == 1
        assert out[0]["name"] == "workflow"
        assert out[0]["content"] == "a\nb"
    finally:
        a.close()


def test_retrieve_http_error_degrades_to_empty_list():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        out = a.retrieve("anything")
        assert out == []  # MUST degrade, not raise (PRD D5)
    finally:
        a.close()


def test_retrieve_network_error_degrades_to_empty_list():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conn refused")
    a = _make_adapter_with_mock("legacy", httpx.MockTransport(handler))
    try:
        out = a.retrieve("anything")
        assert out == []
    finally:
        a.close()


def test_retrieve_handles_bare_list_payload():
    """Some MIRIX endpoints return a bare list (not wrapped in {skills: ...})."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            return _user_resolve_response()
        return httpx.Response(200, json=[
            {"name": "s1", "description": "d1", "instructions": "c1"},
        ])
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        out = a.retrieve("x")
        assert len(out) == 1
        assert out[0]["name"] == "s1"
    finally:
        a.close()


# -- format_for_conversation ------------------------------------------------

def test_format_for_conversation_empty_returns_empty_string():
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://x", user_id="u")
    try:
        assert a.format_for_conversation([]) == ""
    finally:
        a.close()


def test_format_for_conversation_renders_active_skills_block():
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://x", user_id="u")
    try:
        skills = [
            {"name": "iso8601-fix", "description": "fix dates", "content": "use +08:00"},
        ]
        out = a.format_for_conversation(skills)
        assert "## Active Skills" in out
        assert "### iso8601-fix" in out
        assert "_fix dates_" in out
        assert "use +08:00" in out
    finally:
        a.close()


# -- add_skill / add_skills (evolve writeback) ------------------------------

def test_skill_evolve_add_skill_posts_correct_payload():
    captured = {}
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            return _user_resolve_response()
        captured["url"] = req.url.path
        captured["body"] = httpx.Response(200, content=req.content).json()
        return httpx.Response(201, json={"id": "skill_xxx", "name": "iso-fix"})
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        ok = a.add_skill({"name": "iso-fix", "description": "d", "content": "c", "category": "workflow"})
        assert ok is True
        assert captured["url"] == "/v1/skills"
        assert captured["body"]["name"] == "iso-fix"
        assert captured["body"]["instructions"] == "c"
        assert captured["body"]["entry_type"] == "workflow"
        assert captured["body"]["user_id"] == "user-resolved-1234"
    finally:
        a.close()


def test_skill_evolve_add_skill_409_dup_returns_false():
    """Server returns 409 if a skill with the same name already exists;
    adapter must treat this as duplicate and return False without raising."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/users/create_or_get":
            return _user_resolve_response()
        return httpx.Response(409, json={"detail": "Skill with name 'x' already exists"})
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        # Clear the internal dedup cache so the request actually fires
        a._added_names = set()
        ok = a.add_skill({"name": "x", "description": "d", "content": "c"})
        assert ok is False
    finally:
        a.close()


def test_add_skill_duplicate_name_returns_false():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        assert a.add_skill({"name": "x", "description": "d", "content": "c"}) is True
        assert a.add_skill({"name": "x", "description": "d", "content": "c"}) is False
    finally:
        a.close()


def test_add_skill_missing_name_returns_false():
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://x", user_id="u")
    try:
        assert a.add_skill({"description": "d", "content": "c"}) is False
    finally:
        a.close()


def test_add_skills_increments_generation():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        assert a.generation == 0
        a.add_skills([
            {"name": "a", "description": "d", "content": "c"},
            {"name": "b", "description": "d", "content": "c"},
        ])
        # both added → generation incremented once (paper semantics)
        assert a.generation == 1
        a.add_skills([])  # no-op
        assert a.generation == 1
    finally:
        a.close()


def test_add_skill_http_error_returns_false():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})
    a = _make_adapter_with_mock("skill-evolve", httpx.MockTransport(handler))
    try:
        assert a.add_skill({"name": "x", "description": "d", "content": "c"}) is False
    finally:
        a.close()


# -- legacy variant meta-agent bootstrap ------------------------------------

def test_legacy_add_skill_resolves_meta_agent_id():
    calls = []
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/agents":
            return httpx.Response(200, json={
                "agents": [
                    {"id": "agent_42", "name": "meta_memory_agent"},
                    {"id": "agent_99", "name": "other"},
                ],
            })
        if req.url.path == "/memory/add_sync":
            body = httpx.Response(200, content=req.content).json()
            assert body["meta_agent_id"] == "agent_42"
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)
    a = _make_adapter_with_mock("legacy", httpx.MockTransport(handler))
    try:
        ok = a.add_skill({"name": "p", "description": "d", "content": "c"})
        assert ok is True
        assert "/agents" in calls
        assert "/memory/add_sync" in calls
    finally:
        a.close()


# -- get_skill_count --------------------------------------------------------

def test_get_skill_count_initially_zero():
    a = MirixSkillsAdapter(variant="skill-evolve", base_url="http://x", user_id="u")
    try:
        c = a.get_skill_count()
        assert c == {"general": 0, "task_specific": 0, "common_mistakes": 0, "total": 0}
    finally:
        a.close()
