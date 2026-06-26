"""Unit tests for search-result projection helpers.

Every derived-memory search result must carry its ``filter_tags`` (and core
blocks their full ``filter_tags`` alongside the derived ``scope``), for all
memory types, independent of search method, in both single-user and all-users
search. The value lives on the schema object and was previously dropped by the
hand-built result dicts; these helpers are the single projection point.
"""

from types import SimpleNamespace

from mirix.server.rest_api import (
    _project_core_block,
    _project_episodic,
    _project_knowledge,
    _project_procedural,
    _project_resource,
    _project_semantic,
)

_TAGS = {"env": ["staging"], "scope": "team-1"}


def _obj(**kw):
    return SimpleNamespace(**kw)


class TestDerivedProjectionIncludesFilterTags:
    def test_episodic(self):
        x = _obj(
            id="ep-1",
            occurred_at=None,
            event_type="user_message",
            actor="user",
            summary="s",
            details="d",
            filter_tags=_TAGS,
        )
        assert _project_episodic(x)["filter_tags"] == _TAGS

    def test_resource(self):
        x = _obj(
            id="re-1",
            resource_type="markdown",
            title="t",
            summary="s",
            content="c",
            filter_tags=_TAGS,
        )
        assert _project_resource(x)["filter_tags"] == _TAGS

    def test_procedural(self):
        x = _obj(
            id="pr-1",
            entry_type="workflow",
            summary="s",
            steps=["a"],
            filter_tags=_TAGS,
        )
        assert _project_procedural(x)["filter_tags"] == _TAGS

    def test_knowledge_vault(self):
        x = _obj(
            id="kv-1",
            entry_type="bookmark",
            source="src",
            sensitivity="low",
            secret_value="v",
            caption="c",
            filter_tags=_TAGS,
        )
        assert _project_knowledge(x)["filter_tags"] == _TAGS

    def test_semantic(self):
        x = _obj(id="se-1", name="n", summary="s", details="d", source="src", filter_tags=_TAGS)
        out = _project_semantic(x)
        assert out["filter_tags"] == _TAGS
        assert out["name"] == "n"
        assert out["source"] == "src"

    def test_core_block_keeps_scope_and_adds_filter_tags(self):
        block = _obj(id="bl-1", user_id="u1", label="human", value="v", filter_tags=_TAGS)
        out = _project_core_block(block)
        assert out["filter_tags"] == _TAGS
        assert out["scope"] == "team-1"  # derived scope still present

    def test_missing_filter_tags_is_none(self):
        x = _obj(id="se-1", name="n", summary="s", details="d", source="src", filter_tags=None)
        assert _project_semantic(x)["filter_tags"] is None


class TestAllUsersVariantIncludesUserId:
    def test_episodic_all_users_has_user_id_and_filter_tags(self):
        x = _obj(
            id="ep-1",
            occurred_at=None,
            event_type="user_message",
            actor="user",
            summary="s",
            details="d",
            filter_tags=_TAGS,
            user_id="u-9",
        )
        out = _project_episodic(x, include_user_id=True)
        assert out["filter_tags"] == _TAGS
        assert out["user_id"] == "u-9"

    def test_single_user_variant_omits_user_id(self):
        x = _obj(id="se-1", name="n", summary="s", details="d", source="src", filter_tags=_TAGS, user_id="u-9")
        assert "user_id" not in _project_semantic(x)
