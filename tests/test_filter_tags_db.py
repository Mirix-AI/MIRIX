"""
Database-level tests for filter_tags operator queries.

Requires docker PG (port 5433), no running server needed.
Tests write rows with various filter_tags shapes and query using
$contains, $exists, $in operators via the RawMemoryManager against real PG.

Run:
    pytest tests/test_filter_tags_db.py -v
"""

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
]

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.raw_memory import RawMemoryItemCreate
from mirix.schemas.user import User as PydanticUser
from mirix.services.raw_memory_manager import RawMemoryManager


# =================================================================
# FIXTURES
# =================================================================

@pytest.fixture
def raw_memory_manager():
    return RawMemoryManager()


@pytest.fixture(scope="module")
def test_actor():
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.client_manager import ClientManager
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    org_id = f"test-filter-tags-org-{uuid.uuid4().hex[:8]}"
    try:
        org_mgr.get_organization_by_id(org_id)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=org_id, name="Filter Tags Test Org"))

    client_id = f"test-filter-tags-client-{uuid.uuid4().hex[:8]}"
    try:
        return client_mgr.get_client_by_id(client_id)
    except Exception:
        return client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="Filter Tags Test Client",
                write_scope="test-ft",
                read_scopes=["test-ft"],
            )
        )


@pytest.fixture(scope="module")
def test_user(test_actor):
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()
    user_id = f"test-filter-tags-user-{uuid.uuid4().hex[:8]}"
    try:
        return user_mgr.get_user_by_id(user_id)
    except Exception:
        return user_mgr.create_user(
            PydanticUser(
                id=user_id,
                name="Filter Tags Test User",
                organization_id=test_actor.organization_id,
                timezone="UTC",
            )
        )


def _create_memory(raw_memory_manager, test_actor, test_user, context, filter_tags):
    return raw_memory_manager.create_raw_memory(
        raw_memory=RawMemoryItemCreate(
            context=context,
            filter_tags=filter_tags,
            user_id=test_user.id,
            organization_id=test_actor.organization_id,
            occurred_at=None,
            id=None,
            context_embedding=None,
            embedding_config=None,
        ),
        actor=test_actor,
        client_id=test_actor.id,
        user_id=test_user.id,
        use_cache=False,
    )


# =================================================================
# $contains operator
# =================================================================

class TestContainsOperator:
    def test_contains_matches_array_value(self, raw_memory_manager, test_actor, test_user):
        """$contains finds a value inside a stored JSON array."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "contains-match",
            {"scope": "test-ft", "account_ids": ["ABC", "DEF"]},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"account_ids": {"$contains": "ABC"}},
                limit=50,
            )
            assert any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_contains_no_match(self, raw_memory_manager, test_actor, test_user):
        """$contains returns nothing when value is not in the array."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "contains-no-match",
            {"scope": "test-ft", "account_ids": ["ABC", "DEF"]},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"account_ids": {"$contains": "XYZ"}},
                limit=50,
            )
            assert not any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_contains_missing_key_no_error(self, raw_memory_manager, test_actor, test_user):
        """$contains on a key that doesn't exist silently excludes the row."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "contains-missing-key",
            {"scope": "test-ft"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"account_ids": {"$contains": "ABC"}},
                limit=50,
            )
            assert not any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_contains_scalar_value_no_error(self, raw_memory_manager, test_actor, test_user):
        """$contains on a key that holds a scalar (not array) silently excludes the row."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "contains-scalar",
            {"scope": "test-ft", "account_ids": "ABC"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"account_ids": {"$contains": "ABC"}},
                limit=50,
            )
            assert not any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)


# =================================================================
# $exists operator
# =================================================================

class TestExistsOperator:
    def test_exists_true_matches(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "exists-true",
            {"scope": "test-ft", "project_id": "proj-1"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"project_id": {"$exists": True}},
                limit=50,
            )
            assert any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_exists_true_excludes_missing_key(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "exists-true-missing",
            {"scope": "test-ft"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"project_id": {"$exists": True}},
                limit=50,
            )
            assert not any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_exists_false_matches_missing_key(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "exists-false",
            {"scope": "test-ft"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"project_id": {"$exists": False}},
                limit=50,
            )
            assert any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)


# =================================================================
# $in operator
# =================================================================

class TestInOperator:
    def test_in_matches(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "in-match",
            {"scope": "test-ft", "status": "active"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"status": {"$in": ["active", "pending"]}},
                limit=50,
            )
            assert any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_in_no_match(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "in-no-match",
            {"scope": "test-ft", "status": "archived"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"status": {"$in": ["active", "pending"]}},
                limit=50,
            )
            assert not any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)


# =================================================================
# scopes parameter
# =================================================================

class TestScopes:
    def test_scopes_filters_by_scope(self, raw_memory_manager, test_actor, test_user):
        """scopes parameter translates to scope IN (...) correctly.

        create_raw_memory always sets filter_tags.scope = actor.write_scope,
        so every memory here gets scope='test-ft'.  We verify that searching
        with the matching scope finds the memory and a non-matching scope does not.
        """
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "scope-match",
            {},
        )
        try:
            # Matching scope — should find the memory
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                scopes=["test-ft"],
                limit=50,
            )
            result_ids = {r.id for r in results}
            assert mem.id in result_ids

            # Non-matching scope — should NOT find the memory
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                scopes=["other-scope"],
                limit=50,
            )
            result_ids = {r.id for r in results}
            assert mem.id not in result_ids
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_empty_scopes_returns_nothing(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "empty-scopes",
            {"scope": "test-ft"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                scopes=[],
                limit=50,
            )
            assert len(results) == 0
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_read_scopes_in_filter_tags_ignored(self, raw_memory_manager, test_actor, test_user):
        """read_scopes key in filter_tags is ignored; use scopes param instead."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "ignored-read-scopes",
            {"scope": "test-ft"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"read_scopes": ["test-ft"]},
                limit=50,
            )
            result_ids = {r.id for r in results}
            assert mem.id in result_ids
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)


# =================================================================
# Backward compatibility
# =================================================================

class TestBackwardCompatibility:
    def test_plain_scalar_exact_match(self, raw_memory_manager, test_actor, test_user):
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "scalar-match",
            {"scope": "test-ft", "priority": "high"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"priority": "high"},
                limit=50,
            )
            assert any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)

    def test_null_filter_tags_excluded_by_exists(self, raw_memory_manager, test_actor, test_user):
        """Rows with NULL filter_tags are silently excluded by $exists: true."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "null-filter-tags",
            None,
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={"anything": {"$exists": True}},
                limit=50,
            )
            assert not any(r.id == mem.id for r in results)
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)


# =================================================================
# Mixed operators
# =================================================================

class TestMixedOperators:
    def test_contains_and_scalar_combined(self, raw_memory_manager, test_actor, test_user):
        """Combining $contains with a plain scalar filter (AND)."""
        mem = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "mixed-match",
            {"scope": "test-ft", "account_ids": ["ABC", "DEF"], "priority": "high"},
        )
        mem_no_match = _create_memory(
            raw_memory_manager, test_actor, test_user,
            "mixed-no-match",
            {"scope": "test-ft", "account_ids": ["ABC", "DEF"], "priority": "low"},
        )
        try:
            results, _ = raw_memory_manager.search_raw_memories(
                organization_id=test_actor.organization_id,
                user_id=test_user.id,
                filter_tags={
                    "account_ids": {"$contains": "ABC"},
                    "priority": "high",
                },
                limit=50,
            )
            result_ids = {r.id for r in results}
            assert mem.id in result_ids
            assert mem_no_match.id not in result_ids
        finally:
            raw_memory_manager.delete_raw_memory(mem.id, test_actor)
            raw_memory_manager.delete_raw_memory(mem_no_match.id, test_actor)
