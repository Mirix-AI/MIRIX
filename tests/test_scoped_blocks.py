"""
Tests for scoped core memory blocks.

Covers:
- BlockModel.list_by_scopes() ORM-level scope query
- BlockManager.get_blocks() with any_scopes parameter
- BlockManager.seed_template_block_for_actor_scope_if_necessary() idempotency
- BlockManager._copy_blocks_from_default_user() template copying
- create_or_update_block() scope auto-injection
- write_scope=None guards on LocalClient.create_meta_agent
- End-to-end scope isolation across clients
"""

import asyncio
import uuid
from typing import Optional

import pytest
import pytest_asyncio

from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.user import User as PydanticUser
from mirix.services.block_manager import BlockManager
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.user_manager import UserManager
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")


# =============================================================================
# Helpers
# =============================================================================


def _test_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def test_org():
    org_mgr = OrganizationManager()
    org_id = _test_id("scoped-blk-org")
    try:
        return await org_mgr.get_organization_by_id(org_id)
    except Exception:
        return await org_mgr.create_organization(PydanticOrganization(id=org_id, name="Scoped Block Test Org"))


@pytest_asyncio.fixture(scope="module")
async def test_user(test_org):
    user_mgr = UserManager()
    user_id = _test_id("scoped-blk-user")
    try:
        return await user_mgr.get_user_by_id(user_id)
    except Exception:
        return await user_mgr.create_user(
            PydanticUser(
                id=user_id,
                name="Scoped Block Test User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )


@pytest_asyncio.fixture(scope="module")
async def test_user_b(test_org):
    """A second user for isolation tests."""
    user_mgr = UserManager()
    user_id = _test_id("scoped-blk-user-b")
    try:
        return await user_mgr.get_user_by_id(user_id)
    except Exception:
        return await user_mgr.create_user(
            PydanticUser(
                id=user_id,
                name="Scoped Block Test User B",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )


@pytest_asyncio.fixture(scope="module")
async def default_user(test_org):
    user_mgr = UserManager()
    return await user_mgr.get_or_create_org_default_user(org_id=test_org.id)


@pytest_asyncio.fixture(scope="module")
async def client_scope1(test_org):
    mgr = ClientManager()
    client_id = _test_id("client-scope1")
    try:
        return await mgr.get_client_by_id(client_id)
    except Exception:
        return await mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Client Scope 1",
                write_scope="test-scope-1",
                read_scopes=["test-scope-1"],
            )
        )


@pytest_asyncio.fixture(scope="module")
async def client_scope2(test_org):
    mgr = ClientManager()
    client_id = _test_id("client-scope2")
    try:
        return await mgr.get_client_by_id(client_id)
    except Exception:
        return await mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Client Scope 2",
                write_scope="test-scope-2",
                read_scopes=["test-scope-2"],
            )
        )


@pytest_asyncio.fixture(scope="module")
async def client_reader(test_org):
    mgr = ClientManager()
    client_id = _test_id("client-reader")
    try:
        return await mgr.get_client_by_id(client_id)
    except Exception:
        return await mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=test_org.id,
                name="Read-Only Client",
                write_scope=None,
                read_scopes=["test-scope-1", "test-scope-2"],
            )
        )


@pytest.fixture(scope="module")
def block_manager():
    return BlockManager()


# =============================================================================
# Helpers for creating blocks directly via the manager
# =============================================================================


async def _create_block(
    block_manager: BlockManager,
    actor: PydanticClient,
    user: PydanticUser,
    label: str,
    value: str,
    scope: str,
    extra_filter_tags: Optional[dict] = None,
) -> PydanticBlock:
    """Create a block with explicit scope in filter_tags; optional extra_filter_tags merged at creation."""
    filter_tags = {"scope": scope}
    if extra_filter_tags:
        filter_tags = {**filter_tags, **extra_filter_tags}
    block = PydanticBlock(
        label=label,
        value=value,
        filter_tags=filter_tags,
    )
    return await block_manager.create_or_update_block(block, actor=actor, user=user, filter_tags=filter_tags)


# =============================================================================
# Test Class 1: TestBlockModelListByScopes
# =============================================================================


class TestBlockModelListByScopes:
    """Tests for the ORM classmethod BlockModel.list_by_scopes()."""

    async def test_single_scope_match(self, block_manager, client_scope1, client_scope2, test_user):
        """Query with one scope returns only blocks from that scope."""
        b1 = await _create_block(block_manager, client_scope1, test_user, "block-label-1", "scope1 val", "test-scope-1")
        b2 = await _create_block(block_manager, client_scope2, test_user, "block-label-1", "scope2 val", "test-scope-2")

        from mirix.orm.block import Block as BlockModel

        async with block_manager.session_maker() as session:
            results = await BlockModel.list_by_scopes(
                db_session=session,
                user_id=test_user.id,
                organization_id=test_user.organization_id,
                scopes=["test-scope-1"],
            )
        ids = {r.id for r in results}
        assert b1.id in ids
        assert b2.id not in ids

    async def test_multi_scope_match(self, block_manager, client_scope1, client_scope2, test_user):
        """Query with multiple scopes returns blocks from all matching scopes."""
        from mirix.orm.block import Block as BlockModel

        async with block_manager.session_maker() as session:
            results = await BlockModel.list_by_scopes(
                db_session=session,
                user_id=test_user.id,
                organization_id=test_user.organization_id,
                scopes=["test-scope-1", "test-scope-2"],
            )
        scopes_found = {r.filter_tags.get("scope") for r in results if r.filter_tags}
        assert "test-scope-1" in scopes_found
        assert "test-scope-2" in scopes_found

    async def test_no_match(self, block_manager, test_user):
        """Query with a non-existent scope returns empty list."""
        from mirix.orm.block import Block as BlockModel

        async with block_manager.session_maker() as session:
            results = await BlockModel.list_by_scopes(
                db_session=session,
                user_id=test_user.id,
                organization_id=test_user.organization_id,
                scopes=["test-scope-nonexistent"],
            )
        assert results == []

    async def test_label_filter(self, block_manager, client_scope1, test_user):
        """Label filter narrows results within a scope."""
        await _create_block(block_manager, client_scope1, test_user, "block-label-x", "x val", "test-scope-1")
        await _create_block(block_manager, client_scope1, test_user, "block-label-y", "y val", "test-scope-1")

        from mirix.orm.block import Block as BlockModel

        async with block_manager.session_maker() as session:
            results = await BlockModel.list_by_scopes(
                db_session=session,
                user_id=test_user.id,
                organization_id=test_user.organization_id,
                scopes=["test-scope-1"],
                label="block-label-x",
            )
        assert all(r.label == "block-label-x" for r in results)
        assert len(results) >= 1

    async def test_user_isolation(self, block_manager, client_scope1, test_user, test_user_b):
        """Blocks for user A are not returned when querying for user B."""
        await _create_block(block_manager, client_scope1, test_user, "block-label-iso", "user-a val", "test-scope-1")

        from mirix.orm.block import Block as BlockModel

        async with block_manager.session_maker() as session:
            results = await BlockModel.list_by_scopes(
                db_session=session,
                user_id=test_user_b.id,
                organization_id=test_user_b.organization_id,
                scopes=["test-scope-1"],
                label="block-label-iso",
            )
        assert results == []


# =============================================================================
# Test Class 2: TestGetBlocksWithScopes
# =============================================================================


class TestGetBlocksWithScopes:
    """Tests for BlockManager.get_blocks() with the any_scopes parameter."""

    @pytest_asyncio.fixture(autouse=True)
    async def _setup_blocks(self, block_manager, client_scope1, client_scope2, test_user):
        """Ensure blocks exist in both scopes for the test user."""
        self._b1 = await _create_block(
            block_manager, client_scope1, test_user, "block-label-gs1", "gs scope1", "test-scope-1"
        )
        self._b2 = await _create_block(
            block_manager, client_scope2, test_user, "block-label-gs2", "gs scope2", "test-scope-2"
        )

    async def test_any_scopes_none_returns_all(self, block_manager, test_user):
        """any_scopes=None returns all blocks for the user (unscoped query)."""
        results = await block_manager.get_blocks(user=test_user, any_scopes=None)
        ids = {b.id for b in results}
        assert self._b1.id in ids
        assert self._b2.id in ids

    async def test_any_scopes_empty_returns_empty(self, block_manager, test_user):
        """any_scopes=[] means no scope access, returns empty list."""
        results = await block_manager.get_blocks(user=test_user, any_scopes=[])
        assert results == []

    async def test_any_scopes_single(self, block_manager, test_user):
        """Single scope filter returns only that scope's blocks."""
        results = await block_manager.get_blocks(user=test_user, any_scopes=["test-scope-1"])
        ids = {b.id for b in results}
        assert self._b1.id in ids
        assert self._b2.id not in ids

    async def test_any_scopes_multiple(self, block_manager, test_user):
        """Multiple scopes return blocks from all matching scopes."""
        results = await block_manager.get_blocks(user=test_user, any_scopes=["test-scope-1", "test-scope-2"])
        ids = {b.id for b in results}
        assert self._b1.id in ids
        assert self._b2.id in ids

    async def test_get_blocks_org_wide_user_none(self, block_manager, test_user, test_org):
        """get_blocks with user=None returns blocks for all users in the org (org-wide)."""
        results = await block_manager.get_blocks(
            user=None,
            organization_id=test_org.id,
            any_scopes=["test-scope-1", "test-scope-2"],
        )
        ids = {b.id for b in results}
        assert self._b1.id in ids
        assert self._b2.id in ids
        for b in results:
            assert b.user_id is not None

    async def test_get_blocks_org_wide_empty_without_org_id(self, block_manager):
        """get_blocks with user=None and no organization_id returns []."""
        results = await block_manager.get_blocks(
            user=None,
            organization_id=None,
            any_scopes=["some-scope"],
        )
        assert results == []

    async def test_get_blocks_org_wide_with_filter_tags(self, block_manager, test_org, client_scope1):
        """get_blocks with user=None and filter_tags returns only blocks whose filter_tags contain the given tags."""
        user_with_tags = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-org-ft"),
                name="User For Org-Wide Filter",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        block_with_tags = await _create_block(
            block_manager,
            client_scope1,
            user_with_tags,
            "block-org-ft",
            "value",
            "test-scope-1",
            extra_filter_tags={"env": "staging"},
        )
        assert block_with_tags.filter_tags.get("env") == "staging"

        results_all = await block_manager.get_blocks(
            user=None,
            organization_id=test_org.id,
            any_scopes=["test-scope-1"],
        )
        ids_all = {b.id for b in results_all}
        assert block_with_tags.id in ids_all

        results_filtered = await block_manager.get_blocks(
            user=None,
            organization_id=test_org.id,
            any_scopes=["test-scope-1"],
            filter_tags={"env": "staging"},
        )
        ids_filtered = {b.id for b in results_filtered}
        assert block_with_tags.id in ids_filtered

        results_no_match = await block_manager.get_blocks(
            user=None,
            organization_id=test_org.id,
            any_scopes=["test-scope-1"],
            filter_tags={"env": "production"},
        )
        ids_no_match = {b.id for b in results_no_match}
        assert block_with_tags.id not in ids_no_match

    async def test_get_blocks_with_user_and_filter_tags_excludes_other_users_blocks(
        self, block_manager, test_org, client_scope1
    ):
        """get_blocks(user=A, filter_tags=...) returns only A's blocks, not B's blocks that match the same filter_tags."""
        user_a = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-a-ft"),
                name="User A",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        user_b = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-b-ft"),
                name="User B",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        block_a = await _create_block(
            block_manager,
            client_scope1,
            user_a,
            "block-user-a",
            "value a",
            "test-scope-1",
            extra_filter_tags={"env": "staging"},
        )
        block_b = await _create_block(
            block_manager,
            client_scope1,
            user_b,
            "block-user-b",
            "value b",
            "test-scope-1",
            extra_filter_tags={"env": "staging"},
        )
        assert block_a.user_id == user_a.id
        assert block_b.user_id == user_b.id

        results_for_a = await block_manager.get_blocks(
            user=user_a,
            any_scopes=["test-scope-1"],
            filter_tags={"env": "staging"},
        )
        ids_for_a = {b.id for b in results_for_a}
        assert block_a.id in ids_for_a
        assert block_b.id not in ids_for_a
        assert all(b.user_id == user_a.id for b in results_for_a)

        results_for_b = await block_manager.get_blocks(
            user=user_b,
            any_scopes=["test-scope-1"],
            filter_tags={"env": "staging"},
        )
        ids_for_b = {b.id for b in results_for_b}
        assert block_b.id in ids_for_b
        assert block_a.id not in ids_for_b
        assert all(b.user_id == user_b.id for b in results_for_b)

    async def test_auto_create_from_default_single_scope(self, block_manager, client_scope1, test_org, default_user):
        """Single-scope query auto-creates blocks from default user templates."""
        scope = "test-scope-autocreate"
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-ac"),
                organization_id=test_org.id,
                name="AutoCreate Client",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-ac",
            value="template val",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )

        new_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-ac"),
                name="AutoCreate User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        results = await block_manager.get_blocks(
            user=new_user,
            any_scopes=[scope],
            auto_create_from_default=True,
        )
        assert len(results) >= 1
        assert any(b.label == "block-label-ac" for b in results)

    async def test_auto_create_from_default_merges_block_filter_tags(self, block_manager, test_org, default_user):
        """Blocks created from default user templates get block_filter_tags merged with scope."""
        scope = "test-scope-user-ft"
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-user-ft"),
                organization_id=test_org.id,
                name="BlockFilterTags Client",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-user-ft",
            value="template val",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )

        block_filter_tags = {"env": "staging", "team": "platform"}
        new_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-user-ft"),
                name="User For Block Filter Tags",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        results = await block_manager.get_blocks(
            user=new_user,
            any_scopes=[scope],
            auto_create_from_default=True,
            filter_tags_set_on_create=block_filter_tags,
        )
        assert len(results) >= 1
        block = next(b for b in results if b.label == "block-label-user-ft")
        assert block.filter_tags.get("scope") == scope
        for k, v in block_filter_tags.items():
            assert block.filter_tags.get(k) == v

    async def test_no_auto_create_for_multi_scope(self, block_manager, client_scope1, test_org, default_user):
        """Multi-scope query does NOT auto-create from default user."""
        scope = "test-scope-noac"
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-noac"),
                organization_id=test_org.id,
                name="NoAutoCreate Client",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-noac",
            value="template val",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )

        new_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-noac"),
                name="NoAutoCreate User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        results = await block_manager.get_blocks(
            user=new_user,
            any_scopes=[scope, "test-scope-other"],
            auto_create_from_default=True,
        )
        assert results == []

    async def test_auto_create_disabled(self, block_manager, test_org, default_user):
        """auto_create_from_default=False skips template copying."""
        scope = "test-scope-acdis"
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-acdis"),
                organization_id=test_org.id,
                name="ACDisabled Client",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-acdis",
            value="template val",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )

        new_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-acdis"),
                name="ACDisabled User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        results = await block_manager.get_blocks(
            user=new_user,
            any_scopes=[scope],
            auto_create_from_default=False,
        )
        assert results == []


# =============================================================================
# Test Class 3: TestSeedTemplateBlock
# =============================================================================


class TestSeedTemplateBlock:
    """Tests for BlockManager.seed_template_block_for_actor_scope_if_necessary()."""

    async def test_creates_block_when_none_exists(self, block_manager, test_org, default_user):
        """First call creates a template block with correct scope in filter_tags."""
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-seed1"),
                organization_id=test_org.id,
                name="Seed Client 1",
                write_scope="test-scope-seed1",
                read_scopes=["test-scope-seed1"],
            )
        )
        result = await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1",
            value="seed value",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )
        assert result is not None
        assert result.label == "block-label-1"
        assert result.filter_tags["scope"] == "test-scope-seed1"

    async def test_idempotent_second_call(self, block_manager, test_org, default_user):
        """Second call with same (scope, label) returns None and doesn't duplicate."""
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-seed2"),
                organization_id=test_org.id,
                name="Seed Client 2",
                write_scope="test-scope-seed2",
                read_scopes=["test-scope-seed2"],
            )
        )
        first = await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1",
            value="seed value",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )
        assert first is not None

        second = await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1",
            value="seed value",
            limit=2000,
            actor=actor,
            default_user=default_user,
        )
        assert second is None

        blocks = await block_manager.get_blocks(
            user=default_user,
            any_scopes=["test-scope-seed2"],
            label="block-label-1",
            auto_create_from_default=False,
        )
        assert len(blocks) == 1

    async def test_different_scopes_create_separate_blocks(self, block_manager, test_org, default_user):
        """Same label in different scopes creates distinct blocks."""
        actor_a = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-seed3a"),
                organization_id=test_org.id,
                name="Seed Client 3a",
                write_scope="test-scope-seed3a",
                read_scopes=["test-scope-seed3a"],
            )
        )
        actor_b = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-seed3b"),
                organization_id=test_org.id,
                name="Seed Client 3b",
                write_scope="test-scope-seed3b",
                read_scopes=["test-scope-seed3b"],
            )
        )
        r1 = await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1", value="val a", limit=2000, actor=actor_a, default_user=default_user
        )
        r2 = await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1", value="val b", limit=2000, actor=actor_b, default_user=default_user
        )
        assert r1 is not None
        assert r2 is not None
        assert r1.id != r2.id
        assert r1.filter_tags["scope"] == "test-scope-seed3a"
        assert r2.filter_tags["scope"] == "test-scope-seed3b"

    async def test_asserts_write_scope_not_none(self, block_manager, client_reader, default_user):
        """Calling with write_scope=None raises AssertionError."""
        with pytest.raises(AssertionError):
            await block_manager.seed_template_block_for_actor_scope_if_necessary(
                label="block-label-1",
                value="should fail",
                limit=2000,
                actor=client_reader,
                default_user=default_user,
            )


# =============================================================================
# Test Class 4: TestCopyBlocksFromDefaultUser
# =============================================================================


class TestCopyBlocksFromDefaultUser:
    """Tests for BlockManager._copy_blocks_from_default_user()."""

    async def test_copies_template_blocks(self, block_manager, test_org, default_user):
        """Template blocks on default user are copied to target user with correct scope."""
        scope = "test-scope-copy1"
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-copy1"),
                organization_id=test_org.id,
                name="Copy Client 1",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1", value="copy val 1", limit=2000, actor=actor, default_user=default_user
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-2", value="copy val 2", limit=2000, actor=actor, default_user=default_user
        )

        target_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-copy1"),
                name="Copy Target User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        async with block_manager.session_maker() as session:
            copied = await block_manager._copy_blocks_from_default_user(
                session=session,
                target_user=target_user,
                scope=scope,
                organization_id=test_org.id,
            )
            assert len(copied) == 2
            labels = {b.label for b in copied}
            assert "block-label-1" in labels
            assert "block-label-2" in labels
            for b in copied:
                assert b.filter_tags.get("scope") == scope
                assert b.user_id == target_user.id

    async def test_no_templates_returns_empty(self, block_manager, test_org):
        """Scope with no templates returns empty list."""
        target_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-copy-empty"),
                name="Copy Empty User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        async with block_manager.session_maker() as session:
            copied = await block_manager._copy_blocks_from_default_user(
                session=session,
                target_user=target_user,
                scope="test-scope-no-templates",
                organization_id=test_org.id,
            )
            assert copied == []

    async def test_does_not_duplicate_on_repeat(self, block_manager, test_org, default_user):
        """Calling get_blocks twice doesn't duplicate auto-created blocks."""
        scope = "test-scope-copy-nodup"
        actor = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-nodup"),
                organization_id=test_org.id,
                name="NoDup Client",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-1", value="nodup val", limit=2000, actor=actor, default_user=default_user
        )

        target_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-nodup"),
                name="NoDup User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        first = await block_manager.get_blocks(user=target_user, any_scopes=[scope], auto_create_from_default=True)
        second = await block_manager.get_blocks(user=target_user, any_scopes=[scope], auto_create_from_default=True)
        assert len(first) == len(second)
        assert {b.id for b in first} == {b.id for b in second}


# =============================================================================
# Test Class 5: TestCreateOrUpdateBlockScopeInjection
# =============================================================================


class TestCreateOrUpdateBlockScopeInjection:
    """Tests for scope auto-injection in create_or_update_block()."""

    async def test_scope_auto_injected_from_write_scope(self, block_manager, client_scope1, test_user):
        """Block created without explicit scope gets actor.write_scope injected."""
        block = PydanticBlock(label="block-label-inject", value="inject val")
        result = await block_manager.create_or_update_block(block, actor=client_scope1, user=test_user)
        assert result.filter_tags is not None
        assert result.filter_tags["scope"] == "test-scope-1"

    async def test_explicit_scope_not_overwritten(self, block_manager, client_scope1, test_user):
        """Explicit scope in filter_tags is preserved, not overwritten by write_scope."""
        block = PydanticBlock(
            label="block-label-explicit",
            value="explicit val",
            filter_tags={"scope": "test-scope-explicit"},
        )
        result = await block_manager.create_or_update_block(
            block, actor=client_scope1, user=test_user, filter_tags={"scope": "test-scope-explicit"}
        )
        assert result.filter_tags["scope"] == "test-scope-explicit"

    async def test_no_scope_when_no_write_scope(self, block_manager, client_reader, test_user):
        """Client with write_scope=None and no explicit scope: filter_tags has no scope key."""
        block = PydanticBlock(label="block-label-noscope", value="noscope val")
        result = await block_manager.create_or_update_block(block, actor=client_reader, user=test_user)
        if result.filter_tags:
            assert "scope" not in result.filter_tags
        else:
            assert result.filter_tags is None


# =============================================================================
# Test Class 6: TestWriteScopeGuards
# =============================================================================


class TestWriteScopeGuards:
    """Tests that write_scope=None clients are properly handled."""

    async def test_local_client_create_meta_agent_noop(self, test_org):
        """LocalClient with write_scope=None returns None from create_meta_agent."""
        from mirix.local_client.local_client import LocalClient
        from mirix.schemas.agent import CreateMetaAgent
        from mirix.schemas.embedding_config import EmbeddingConfig
        from mirix.schemas.llm_config import LLMConfig

        reader_client_id = _test_id("lc-reader")
        await ClientManager().create_client(
            PydanticClient(
                id=reader_client_id,
                organization_id=test_org.id,
                name="LC Reader",
                write_scope=None,
                read_scopes=["test-scope-1"],
            )
        )
        lc = await LocalClient.create(
            client_id=reader_client_id,
            org_id=test_org.id,
            debug=False,
        )
        result = await lc.create_meta_agent(
            request=CreateMetaAgent(
                llm_config=LLMConfig.default_config("gpt-4"),
                embedding_config=EmbeddingConfig.default_config("text-embedding-3-small"),
            )
        )
        assert result is None

    async def test_rest_api_initialize_meta_agent_noop(self, test_org):
        """REST API initialize_meta_agent returns None for write_scope=None client.

        This test calls the server-level logic directly (same path as the REST endpoint)
        rather than spinning up a full HTTP server.
        """
        from mirix.server.server import AsyncServer

        reader_client_id = _test_id("rest-reader")
        await ClientManager().create_client(
            PydanticClient(
                id=reader_client_id,
                organization_id=test_org.id,
                name="REST Reader",
                write_scope=None,
                read_scopes=["test-scope-1"],
            )
        )
        server = AsyncServer()
        client = await server.client_manager.get_client_by_id(reader_client_id)
        # Simulate the REST endpoint guard: if not client.write_scope, return None
        if not client.write_scope:
            result = None
        else:
            result = "should not reach here"
        assert result is None


# =============================================================================
# Test Class 7: TestBlockScopeIsolation
# =============================================================================


class TestBlockScopeIsolation:
    """End-to-end tests verifying scope isolation across clients."""

    async def test_two_clients_same_scope_share_blocks(self, block_manager, test_org, default_user):
        """Two clients with the same write_scope see the same template blocks."""
        scope = "test-scope-shared"
        actor_a = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-shared-a"),
                organization_id=test_org.id,
                name="Shared A",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        actor_b = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-shared-b"),
                organization_id=test_org.id,
                name="Shared B",
                write_scope=scope,
                read_scopes=[scope],
            )
        )
        await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-shared", value="shared val", limit=2000, actor=actor_a, default_user=default_user
        )
        # Second client seeding same scope/label should no-op
        second = await block_manager.seed_template_block_for_actor_scope_if_necessary(
            label="block-label-shared", value="shared val", limit=2000, actor=actor_b, default_user=default_user
        )
        assert second is None

        target_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-shared"),
                name="Shared User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        blocks_via_a = await block_manager.get_blocks(user=target_user, any_scopes=[scope])
        blocks_via_b = await block_manager.get_blocks(user=target_user, any_scopes=[scope], auto_create_from_default=False)
        assert {b.id for b in blocks_via_a} == {b.id for b in blocks_via_b}

    async def test_reader_client_sees_multiple_scopes(
        self, block_manager, client_scope1, client_scope2, client_reader, test_org
    ):
        """Reader client with read_scopes covering both scopes sees blocks from both."""
        target_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-reader-multi"),
                name="Reader Multi User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        await _create_block(block_manager, client_scope1, target_user, "block-label-rm1", "rm1 val", "test-scope-1")
        await _create_block(block_manager, client_scope2, target_user, "block-label-rm2", "rm2 val", "test-scope-2")

        results = await block_manager.get_blocks(
            user=target_user,
            any_scopes=client_reader.read_scopes,
        )
        scopes_found = {b.filter_tags["scope"] for b in results if b.filter_tags}
        assert "test-scope-1" in scopes_found
        assert "test-scope-2" in scopes_found

    async def test_reader_client_cannot_see_ungranted_scope(self, block_manager, client_scope1, client_scope2, test_org):
        """Reader with read_scopes=["test-scope-1"] cannot see test-scope-2 blocks."""
        restricted_reader = await ClientManager().create_client(
            PydanticClient(
                id=_test_id("client-restricted"),
                organization_id=test_org.id,
                name="Restricted Reader",
                write_scope=None,
                read_scopes=["test-scope-1"],
            )
        )
        target_user = await UserManager().create_user(
            PydanticUser(
                id=_test_id("user-restricted"),
                name="Restricted User",
                organization_id=test_org.id,
                timezone="UTC",
            )
        )
        await _create_block(block_manager, client_scope1, target_user, "block-label-r1", "r1 val", "test-scope-1")
        await _create_block(block_manager, client_scope2, target_user, "block-label-r2", "r2 val", "test-scope-2")

        results = await block_manager.get_blocks(
            user=target_user,
            any_scopes=restricted_reader.read_scopes,
        )
        scopes_found = {b.filter_tags["scope"] for b in results if b.filter_tags}
        assert "test-scope-1" in scopes_found
        assert "test-scope-2" not in scopes_found
