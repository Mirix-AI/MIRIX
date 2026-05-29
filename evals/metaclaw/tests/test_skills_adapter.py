"""Unit tests for :mod:`evals.metaclaw.mirix_adapters.skills_adapter`.

Uses :class:`httpx.MockTransport` to assert wire-level GET / POST shape
against the MIRIX REST contract without touching the live server.

Covered cases (per issue 03 acceptance criteria):

  1.  ``retrieve()`` issues correct GET (URL, query params, X-Client-Id header)
  2.  ``.skills`` property buckets MIRIX rows into paper's
      ``{general_skills, task_specific_skills, common_mistakes}`` shape
  3.  ``format_for_conversation`` is byte-identical to
      ``metaclaw.skill_manager.SkillManager.format_for_conversation``
  4.  ``add_skill`` maps paper-category → MIRIX entry_type and posts correctly
  5.  HTTP 409 on duplicate add → False
  6.  Round-trip preserves paper category via ``[paper-category=X] `` prefix
"""

from __future__ import annotations

import json
from typing import Callable, List

import httpx
import pytest

from evals.metaclaw.mirix_adapters.skills_adapter import (
    DEFAULT_CLIENT_ID,
    MIRIX_ENTRY_TYPES,
    MirixSkillsAdapter,
    _map_paper_category_to_entry_type,
    _restore_paper_category,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _user_resp(user_id: str) -> httpx.Response:
    """Canned response for POST /users/create_or_get."""
    return httpx.Response(
        200,
        json={
            "id": user_id,
            "organization_id": "org-test",
            "name": user_id,
            "status": "active",
            "timezone": "UTC",
            "is_admin": False,
            "created_at": "2026-05-28T00:00:00Z",
            "updated_at": "2026-05-28T00:00:00Z",
            "is_deleted": False,
        },
    )


def _make_adapter(
    handler: Callable[[httpx.Request], httpx.Response], **kw
) -> MirixSkillsAdapter:
    """Construct an adapter wired to a MockTransport."""
    transport = httpx.MockTransport(handler)
    return MirixSkillsAdapter(
        base_url="http://mock.test",
        user_id=kw.pop("user_id", "u-test"),
        top_k=kw.pop("top_k", 6),
        transport=transport,
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1.  retrieve()                                                                #
# --------------------------------------------------------------------------- #


def test_retrieve_issues_correct_get_with_headers_and_params():
    captured: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        # /v1/skills
        return httpx.Response(
            200,
            json={
                "success": True,
                "skills": [
                    {
                        "id": "p1",
                        "name": "alpha",
                        "description": "a skill",
                        "instructions": "do alpha",
                        "entry_type": "guide",
                    },
                    {
                        "id": "p2",
                        "name": "beta",
                        "description": "another skill",
                        "instructions": "do beta",
                        "entry_type": "workflow",
                    },
                ],
                "total_count": 2,
            },
        )

    a = _make_adapter(handler, top_k=3)
    out = a.retrieve("ship the rocket")
    assert len(out) == 2
    assert out[0] == {
        "name": "alpha",
        "description": "a skill",
        "content": "do alpha",
        "category": "guide",
    }

    # We expect: (1) POST /users/create_or_get, (2) GET /v1/skills.
    assert [r.method + " " + r.url.path for r in captured] == [
        "POST /users/create_or_get",
        "GET /v1/skills",
    ]
    get_req = captured[1]
    assert get_req.url.params["query"] == "ship the rocket"
    assert get_req.url.params["limit"] == "3"
    assert get_req.url.params["user_id"] == "u-test"
    assert get_req.headers["X-Client-Id"] == DEFAULT_CLIENT_ID


def test_retrieve_returns_empty_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(500, text="boom")

    a = _make_adapter(handler)
    assert a.retrieve("anything") == []


def test_retrieve_relevant_is_alias_for_retrieve():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(
            200,
            json={"success": True, "skills": [], "total_count": 0},
        )

    a = _make_adapter(handler)
    # Should not raise even when paper passes min_relevance kwarg.
    assert a.retrieve_relevant("q", top_k=4, min_relevance=0.07) == []


# --------------------------------------------------------------------------- #
# 2.  .skills property buckets correctly                                        #
# --------------------------------------------------------------------------- #


def test_skills_property_buckets_by_category():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        # Returns a mix: 1 general (via prefix), 1 common_mistakes (via prefix),
        # 1 coding (via prefix), 1 raw workflow (no prefix).
        return httpx.Response(
            200,
            json={
                "success": True,
                "skills": [
                    {
                        "name": "skill-a",
                        "description": "[paper-category=general] a generic guide",
                        "instructions": "step1",
                        "entry_type": "guide",
                    },
                    {
                        "name": "skill-b",
                        "description": "[paper-category=common_mistakes] avoid this",
                        "instructions": "step2",
                        "entry_type": "guide",
                    },
                    {
                        "name": "skill-c",
                        "description": "[paper-category=coding] write code",
                        "instructions": "step3",
                        "entry_type": "guide",
                    },
                    {
                        "name": "skill-d",
                        "description": "a workflow",
                        "instructions": "step4",
                        "entry_type": "workflow",
                    },
                ],
                "total_count": 4,
            },
        )

    a = _make_adapter(handler)
    snap = a.skills
    # general bucket
    assert len(snap["general_skills"]) == 1
    assert snap["general_skills"][0]["name"] == "skill-a"
    # common_mistakes bucket
    assert len(snap["common_mistakes"]) == 1
    assert snap["common_mistakes"][0]["name"] == "skill-b"
    # task-specific
    assert set(snap["task_specific_skills"].keys()) == {"coding", "workflow"}
    assert snap["task_specific_skills"]["coding"][0]["name"] == "skill-c"
    assert snap["task_specific_skills"]["workflow"][0]["name"] == "skill-d"


def test_skills_setter_is_noop():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(200, json={"success": True, "skills": []})

    a = _make_adapter(handler)
    # Paper code occasionally does `manager.skills = ...` defensively.  Must not raise.
    a.skills = {"foo": "bar"}
    # And the property still returns a live snapshot, not the assigned value.
    assert a.skills["general_skills"] == []


def test_skills_returns_empty_bank_on_backend_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(500, text="db down")

    a = _make_adapter(handler)
    snap = a.skills
    assert snap == {
        "general_skills": [],
        "task_specific_skills": {},
        "common_mistakes": [],
    }


# --------------------------------------------------------------------------- #
# 3.  format_for_conversation matches paper byte-for-byte                       #
# --------------------------------------------------------------------------- #


def test_format_for_conversation_byte_identical_to_paper():
    # Import the paper reference and compare against the adapter rendering.
    from metaclaw.skill_manager import SkillManager  # vendored

    skills = [
        {"name": "skill-a", "description": "desc-a", "content": "body-a"},
        {"name": "skill-b", "description": "desc-b", "content": ""},
        {"name": "skill-c", "description": "", "content": "body-c"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(200, json={"success": True, "skills": []})

    a = _make_adapter(handler)

    # We don't construct a paper SkillManager (it needs a real dir).  Instead,
    # invoke the unbound static-shape method by binding it to a dummy ``self``.
    paper_render = SkillManager.format_for_conversation(SkillManager, skills)
    adapter_render = a.format_for_conversation(skills)
    assert adapter_render == paper_render
    # Sanity: empty list -> empty string in both.
    assert a.format_for_conversation([]) == ""
    assert SkillManager.format_for_conversation(SkillManager, []) == ""


# --------------------------------------------------------------------------- #
# 4.  add_skill posts correctly and maps category                               #
# --------------------------------------------------------------------------- #


def test_add_skill_posts_and_maps_known_entry_type():
    captured: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "success": True,
                "skill": {
                    "id": "proc_NEW",
                    "name": body["name"],
                    "entry_type": body["entry_type"],
                    "description": body["description"],
                    "instructions": body["instructions"],
                    "version": "0.1.0",
                    "created_at": "2026-05-28T00:00:00Z",
                },
            },
        )

    a = _make_adapter(handler)
    ok = a.add_skill(
        {
            "name": "deploy-thing",
            "description": "deploys it",
            "content": "step 1; step 2",
            "category": "workflow",  # already a valid MIRIX entry_type
        }
    )
    assert ok is True
    post = next(
        r for r in captured if r.method == "POST" and r.url.path == "/v1/skills"
    )
    body = json.loads(post.content)
    assert body["name"] == "deploy-thing"
    assert body["entry_type"] == "workflow"
    # category is a valid MIRIX entry_type, so NO paper-category prefix added.
    assert body["description"] == "deploys it"
    assert body["instructions"] == "step 1; step 2"
    assert body["user_id"] == "u-test"


def test_add_skill_prefixes_description_for_unmappable_category():
    captured: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(200, json={"success": True, "skill": {}})

    a = _make_adapter(handler)
    ok = a.add_skill(
        {
            "name": "be-careful",
            "description": "watch out for foo",
            "content": "don't do x",
            "category": "common_mistakes",
        }
    )
    assert ok is True
    post = next(
        r for r in captured if r.method == "POST" and r.url.path == "/v1/skills"
    )
    body = json.loads(post.content)
    # common_mistakes is not in MIRIX_ENTRY_TYPES -> coerced to "guide" + prefix.
    assert body["entry_type"] == "guide"
    assert body["description"] == "[paper-category=common_mistakes] watch out for foo"


def test_add_skill_missing_name_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="should not be called")

    a = _make_adapter(handler)
    assert a.add_skill({}) is False
    assert a.add_skill({"name": "   "}) is False


def test_add_skills_increments_generation_when_any_added():
    state = {"posts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        state["posts"] += 1
        return httpx.Response(200, json={"success": True, "skill": {}})

    a = _make_adapter(handler)
    assert a.generation == 0
    added = a.add_skills(
        [
            {"name": "s1", "description": "d1", "content": "c1"},
            {"name": "s2", "description": "d2", "content": "c2"},
        ],
        category="general",
    )
    assert added == 2
    assert a.generation == 1
    # Empty batch: no increment.
    assert a.add_skills([], category="general") == 0
    assert a.generation == 1


# --------------------------------------------------------------------------- #
# 5.  HTTP 409 → False                                                          #
# --------------------------------------------------------------------------- #


def test_add_skill_409_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(
            409,
            json={"detail": "Skill with name 'dup' already exists (ID: proc_X)"},
        )

    a = _make_adapter(handler)
    assert a.add_skill({"name": "dup", "description": "d", "content": "c"}) is False


def test_add_skill_500_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(500, text="db oops")

    a = _make_adapter(handler)
    assert a.add_skill({"name": "x", "description": "d", "content": "c"}) is False


# --------------------------------------------------------------------------- #
# 6.  Round-trip preserves paper category via [paper-category=X] prefix         #
# --------------------------------------------------------------------------- #


def test_round_trip_preserves_paper_category():
    """add_skill(category='coding') -> server stores prefixed desc ->
    retrieve() restores category='coding' on read."""
    stored: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        if request.method == "POST" and request.url.path == "/v1/skills":
            body = json.loads(request.content)
            stored["row"] = {
                "id": "proc_RT",
                "name": body["name"],
                "description": body["description"],
                "instructions": body["instructions"],
                "entry_type": body["entry_type"],
            }
            return httpx.Response(200, json={"success": True, "skill": stored["row"]})
        if request.method == "GET" and request.url.path == "/v1/skills":
            rows = [stored["row"]] if stored else []
            return httpx.Response(
                200,
                json={"success": True, "skills": rows, "total_count": len(rows)},
            )
        return httpx.Response(404)

    a = _make_adapter(handler)
    ok = a.add_skill(
        {
            "name": "rt-skill",
            "description": "round-trip me",
            "content": "step",
            "category": "coding",
        }
    )
    assert ok is True
    # Server-side, description was prefixed.
    assert stored["row"]["description"].startswith("[paper-category=coding] ")
    assert stored["row"]["entry_type"] == "guide"

    # Now read back via retrieve() — category should be restored, description clean.
    out = a.retrieve("rt-skill")
    assert len(out) == 1
    assert out[0]["name"] == "rt-skill"
    assert out[0]["category"] == "coding"
    assert out[0]["description"] == "round-trip me"
    assert out[0]["content"] == "step"

    # And via .skills, the row should land in task_specific_skills["coding"].
    snap = a.skills
    assert "coding" in snap["task_specific_skills"]
    assert snap["task_specific_skills"]["coding"][0]["name"] == "rt-skill"


# --------------------------------------------------------------------------- #
# Bonus: category-mapping table is correct                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "paper_cat, expected_entry_type",
    [
        ("guide", "guide"),
        ("script", "script"),
        ("workflow", "workflow"),
        ("WORKFLOW", "workflow"),  # case-insensitive
        ("task", "workflow"),
        ("deployment", "workflow"),
        ("ingestion", "workflow"),
        ("pipeline", "workflow"),
        ("general", "guide"),
        ("common_mistakes", "guide"),
        ("coding", "guide"),
        ("", "guide"),
        (None, "guide"),
    ],
)
def test_map_paper_category_table(paper_cat, expected_entry_type):
    assert _map_paper_category_to_entry_type(paper_cat) == expected_entry_type


def test_restore_paper_category_extracts_and_strips_prefix():
    desc, cat = _restore_paper_category(
        "[paper-category=research] go find papers", "guide"
    )
    assert desc == "go find papers"
    assert cat == "research"


def test_restore_paper_category_passes_through_when_no_prefix():
    desc, cat = _restore_paper_category("regular description", "workflow")
    assert desc == "regular description"
    assert cat == "workflow"


# --------------------------------------------------------------------------- #
# Surface coverage check                                                        #
# --------------------------------------------------------------------------- #


def test_adapter_covers_paper_skillmanager_surface():
    """Every attribute api_server.py reads on SkillManager must exist on the adapter."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "skills": []})

    a = _make_adapter(handler)
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
        "close",
    ):
        assert hasattr(a, attr), f"MirixSkillsAdapter missing attribute: {attr}"


def test_get_skill_count_live_counts():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/create_or_get":
            return _user_resp("u-test")
        return httpx.Response(
            200,
            json={
                "success": True,
                "skills": [
                    {
                        "name": "g1",
                        "description": "[paper-category=general] x",
                        "instructions": "",
                        "entry_type": "guide",
                    },
                    {
                        "name": "g2",
                        "description": "[paper-category=general] y",
                        "instructions": "",
                        "entry_type": "guide",
                    },
                    {
                        "name": "t1",
                        "description": "[paper-category=coding] z",
                        "instructions": "",
                        "entry_type": "guide",
                    },
                    {
                        "name": "m1",
                        "description": "[paper-category=common_mistakes] q",
                        "instructions": "",
                        "entry_type": "guide",
                    },
                ],
            },
        )

    a = _make_adapter(handler)
    counts = a.get_skill_count()
    assert counts == {
        "general": 2,
        "task_specific": 1,
        "common_mistakes": 1,
        "total": 4,
    }


def test_reload_and_save_are_noops():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="should not be hit")

    a = _make_adapter(handler)
    assert a.reload() is None
    assert a.save() is None
    assert a.save(path="/tmp/anything.json") is None


def test_close_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    a = _make_adapter(handler)
    a.close()
    # Second close is also safe.
    a.close()


# --------------------------------------------------------------------------- #
# MIRIX_ENTRY_TYPES sanity                                                      #
# --------------------------------------------------------------------------- #


def test_mirix_entry_types_constant():
    assert MIRIX_ENTRY_TYPES == {"guide", "script", "workflow"}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
