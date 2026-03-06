"""
Tests for block_filter_tags_update_mode feature.

Covers:
- Remote client: update_mode included in request body
- Local client: update_mode forwarded to server.send_messages
- Queue: update_mode serialized/deserialized through protobuf
- Agent._apply_block_filter_tags: merge vs replace logic, no-op skip
- BlockManager.update_block_filter_tags: DB persistence

Run unit tests:
    pytest tests/test_block_filter_tags_update_mode.py -v -k "not integration"

Run all (requires docker PG):
    pytest tests/test_block_filter_tags_update_mode.py -v
"""

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.queue.queue_util import put_messages
from mirix.schemas.client import Client
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import MessageCreate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _test_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Shared fixtures for queue tests
# ---------------------------------------------------------------------------

QUEUE_TEST_ORG_ID = "org-bft-update-mode"


@pytest.fixture
def ensure_queue_org():
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    try:
        org_mgr.get_organization_by_id(QUEUE_TEST_ORG_ID)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=QUEUE_TEST_ORG_ID, name="BFT Queue Test Org"))
    return QUEUE_TEST_ORG_ID


@pytest.fixture
def queue_sample_client(ensure_queue_org):
    return Client(
        id="client-bft-queue",
        organization_id=ensure_queue_org,
        name="BFT Queue Client",
        status="active",
        write_scope="test",
        read_scopes=["test"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


@pytest.fixture
def queue_sample_messages():
    return [MessageCreate(role=MessageRole.user, content="Hello")]


@pytest.fixture
def queue_mock_server():
    server = Mock()
    server.send_messages = Mock(
        return_value=Mock(model_dump=Mock(return_value={"completion_tokens": 10, "prompt_tokens": 5}))
    )
    return server


@pytest.fixture
def queue_clean_manager():
    from mirix.queue.manager import get_manager

    manager = get_manager()
    if manager.is_initialized:
        manager.cleanup()
    return manager


# ============================================================================
# 1. Remote Client Tests (unit, no DB)
# ============================================================================


@pytest.fixture
def remote_client():
    from mirix.client import MirixClient

    with patch.object(MirixClient, "_request") as mock_request:
        mock_request.return_value = {"messages": [], "usage": {}}
        c = MirixClient(api_key="test-key", debug=False)
        yield c


class TestRemoteClientUpdateMode:
    """MirixClient passes block_filter_tags_update_mode in request body."""

    def test_send_message_default_merge_not_sent(self, remote_client):
        """Default 'merge' is not explicitly included (server defaults)."""
        remote_client._request = Mock(return_value={"messages": [], "usage": {}})
        remote_client.send_message(
            message="Hello",
            role="user",
            agent_id="agent-1",
            block_filter_tags={"env": "prod"},
        )
        body = remote_client._request.call_args.kwargs["json"]
        assert "block_filter_tags_update_mode" not in body

    def test_send_message_replace_mode_sent(self, remote_client):
        """When 'replace' is specified, it's included in the request body."""
        remote_client._request = Mock(return_value={"messages": [], "usage": {}})
        remote_client.send_message(
            message="Hello",
            role="user",
            agent_id="agent-1",
            block_filter_tags={"env": "prod"},
            block_filter_tags_update_mode="replace",
        )
        body = remote_client._request.call_args.kwargs["json"]
        assert body["block_filter_tags_update_mode"] == "replace"

    def test_user_message_forwards_update_mode(self, remote_client):
        """user_message forwards block_filter_tags_update_mode to send_message."""
        remote_client._request = Mock(return_value={"messages": [], "usage": {}})
        remote_client.user_message(
            agent_id="agent-2",
            message="Hi",
            block_filter_tags={"env": "staging"},
            block_filter_tags_update_mode="replace",
        )
        body = remote_client._request.call_args.kwargs["json"]
        assert body["block_filter_tags_update_mode"] == "replace"

    def test_add_memory_replace_mode_sent(self, remote_client):
        """add() includes block_filter_tags_update_mode when not default."""
        remote_client._meta_agent = SimpleNamespace(id="meta-1")
        remote_client._ensure_user_exists = Mock()
        remote_client._request = Mock(return_value={"success": True})
        remote_client.add(
            user_id="user-1",
            messages=[{"role": "user", "content": "hi"}],
            block_filter_tags={"team": "core"},
            block_filter_tags_update_mode="replace",
        )
        body = remote_client._request.call_args.kwargs["json"]
        assert body["block_filter_tags_update_mode"] == "replace"

    def test_add_memory_default_merge_not_sent(self, remote_client):
        """add() with default merge does not include update_mode."""
        remote_client._meta_agent = SimpleNamespace(id="meta-1")
        remote_client._ensure_user_exists = Mock()
        remote_client._request = Mock(return_value={"success": True})
        remote_client.add(
            user_id="user-1",
            messages=[{"role": "user", "content": "hi"}],
            block_filter_tags={"team": "core"},
        )
        body = remote_client._request.call_args.kwargs["json"]
        assert "block_filter_tags_update_mode" not in body


# ============================================================================
# 2. Local Client Tests (unit, signature inspection)
# ============================================================================


class TestLocalClientUpdateMode:
    """LocalClient.send_messages accepts and forwards block_filter_tags_update_mode."""

    def test_send_messages_signature_accepts_update_mode(self):
        """The send_messages method accepts block_filter_tags_update_mode."""
        import inspect

        from mirix.local_client.local_client import LocalClient

        sig = inspect.signature(LocalClient.send_messages)
        assert "block_filter_tags_update_mode" in sig.parameters
        param = sig.parameters["block_filter_tags_update_mode"]
        assert param.default == "merge"

    def test_send_message_signature_accepts_update_mode(self):
        """The send_message method accepts block_filter_tags_update_mode."""
        import inspect

        from mirix.local_client.local_client import LocalClient

        sig = inspect.signature(LocalClient.send_message)
        assert "block_filter_tags_update_mode" in sig.parameters
        param = sig.parameters["block_filter_tags_update_mode"]
        assert param.default == "merge"


# ============================================================================
# 3. Agent._apply_block_filter_tags Tests (unit, no DB)
# ============================================================================


def _apply_block_filter_tags_standalone(self_obj, blocks):
    """Standalone copy of Agent._apply_block_filter_tags for testing without heavy imports.

    Must be kept in sync with Agent._apply_block_filter_tags in mirix/agent/agent.py.
    """
    safe_tags = {k: v for k, v in self_obj.block_filter_tags.items() if k != "scope"}
    for block in blocks:
        existing_tags = block.filter_tags or {}
        scope = existing_tags.get("scope")

        if self_obj.block_filter_tags_update_mode == "replace":
            desired = {**safe_tags}
            if scope is not None:
                desired["scope"] = scope
        else:
            desired = {**existing_tags, **safe_tags}

        if desired != existing_tags:
            block.filter_tags = desired
            self_obj.block_manager.update_block_filter_tags(
                block_id=block.id,
                new_filter_tags=desired,
                actor=self_obj.actor,
                user=self_obj.user,
            )
    return blocks


class TestApplyBlockFilterTags:
    """Unit tests for the _apply_block_filter_tags merge/replace logic.

    Uses a standalone copy of the method to avoid importing the full Agent class
    (which pulls in langfuse and other heavy dependencies).
    """

    def _make_agent(self, block_filter_tags, update_mode="merge"):
        agent = SimpleNamespace()
        agent.block_filter_tags = block_filter_tags
        agent.block_filter_tags_update_mode = update_mode
        agent.actor = SimpleNamespace(id="test-client")
        agent.user = None
        agent.block_manager = Mock()
        agent.block_manager.update_block_filter_tags = Mock(return_value=None)
        return agent

    def _make_block(self, filter_tags):
        from mirix.schemas.block import Block as PydanticBlock

        return PydanticBlock(
            id=_test_id("block"),
            label="human",
            value="test value",
            filter_tags=filter_tags,
        )

    def test_merge_adds_new_keys(self):
        agent = self._make_agent({"team": "platform"}, "merge")
        block = self._make_block({"scope": "scope-1", "env": "prod"})
        blocks = [block]

        result = _apply_block_filter_tags_standalone(agent, blocks)

        assert result is blocks
        agent.block_manager.update_block_filter_tags.assert_called_once()
        assert block.filter_tags == {"scope": "scope-1", "env": "prod", "team": "platform"}

    def test_merge_overwrites_existing_keys(self):
        agent = self._make_agent({"env": "staging"}, "merge")
        block = self._make_block({"scope": "scope-1", "env": "prod"})

        _apply_block_filter_tags_standalone(agent, [block])

        assert block.filter_tags["env"] == "staging"
        assert block.filter_tags["scope"] == "scope-1"

    def test_replace_preserves_scope(self):
        agent = self._make_agent({"team": "platform"}, "replace")
        block = self._make_block({"scope": "scope-1", "env": "prod", "old_key": "old_val"})

        _apply_block_filter_tags_standalone(agent, [block])

        assert block.filter_tags["scope"] == "scope-1"
        assert block.filter_tags["team"] == "platform"
        assert "env" not in block.filter_tags
        assert "old_key" not in block.filter_tags

    def test_replace_without_existing_scope(self):
        agent = self._make_agent({"team": "platform"}, "replace")
        block = self._make_block({"env": "prod"})

        _apply_block_filter_tags_standalone(agent, [block])

        assert block.filter_tags == {"team": "platform"}
        assert "scope" not in block.filter_tags

    def test_noop_when_tags_already_match(self):
        """No DB write when desired tags equal existing tags."""
        agent = self._make_agent({"env": "prod"}, "merge")
        block = self._make_block({"scope": "scope-1", "env": "prod"})

        result = _apply_block_filter_tags_standalone(agent, [block])

        agent.block_manager.update_block_filter_tags.assert_not_called()
        assert result[0] is block

    def test_noop_after_creation_with_same_tags(self):
        """Simulates the fresh-creation case: tags already include block_filter_tags."""
        tags = {"scope": "scope-1", "env": "staging", "team": "platform"}
        agent = self._make_agent({"env": "staging", "team": "platform"}, "merge")
        block = self._make_block(tags)

        result = _apply_block_filter_tags_standalone(agent, [block])

        agent.block_manager.update_block_filter_tags.assert_not_called()

    def test_multiple_blocks(self):
        """All blocks in the list are processed."""
        agent = self._make_agent({"team": "new"}, "merge")
        block_a = self._make_block({"scope": "s1", "team": "old"})
        block_b = self._make_block({"scope": "s2"})

        result = _apply_block_filter_tags_standalone(agent, [block_a, block_b])

        assert len(result) == 2
        assert agent.block_manager.update_block_filter_tags.call_count == 2

    def test_empty_block_list(self):
        """Empty list returns empty list, no errors."""
        agent = self._make_agent({"team": "platform"}, "merge")
        result = _apply_block_filter_tags_standalone(agent, [])
        assert result == []
        agent.block_manager.update_block_filter_tags.assert_not_called()

    def test_block_with_none_filter_tags(self):
        """Block with filter_tags=None gets tags applied."""
        agent = self._make_agent({"team": "platform"}, "merge")
        block = self._make_block(None)

        _apply_block_filter_tags_standalone(agent, [block])

        agent.block_manager.update_block_filter_tags.assert_called_once()
        assert block.filter_tags == {"team": "platform"}

    # --- Scope protection tests ---

    def test_merge_cannot_overwrite_scope(self):
        """Merge mode: 'scope' in block_filter_tags is silently stripped."""
        agent = self._make_agent({"scope": "evil-scope", "team": "platform"}, "merge")
        block = self._make_block({"scope": "original-scope", "env": "prod"})

        _apply_block_filter_tags_standalone(agent, [block])

        assert block.filter_tags["scope"] == "original-scope"
        assert block.filter_tags["team"] == "platform"

    def test_replace_cannot_overwrite_scope(self):
        """Replace mode: 'scope' in block_filter_tags is silently stripped; original scope preserved."""
        agent = self._make_agent({"scope": "evil-scope", "team": "platform"}, "replace")
        block = self._make_block({"scope": "original-scope", "env": "prod"})

        _apply_block_filter_tags_standalone(agent, [block])

        assert block.filter_tags["scope"] == "original-scope"
        assert block.filter_tags["team"] == "platform"
        assert "env" not in block.filter_tags

    def test_merge_scope_only_is_noop(self):
        """Merge with only 'scope' in block_filter_tags is a no-op (scope stripped, nothing to merge)."""
        agent = self._make_agent({"scope": "evil-scope"}, "merge")
        block = self._make_block({"scope": "original-scope", "env": "prod"})

        result = _apply_block_filter_tags_standalone(agent, [block])

        agent.block_manager.update_block_filter_tags.assert_not_called()
        assert result[0] is block

    def test_replace_scope_only_preserves_original(self):
        """Replace with only 'scope' in block_filter_tags keeps original scope, drops other tags."""
        agent = self._make_agent({"scope": "evil-scope"}, "replace")
        block = self._make_block({"scope": "original-scope", "env": "prod", "team": "old"})

        _apply_block_filter_tags_standalone(agent, [block])

        assert block.filter_tags == {"scope": "original-scope"}


# ============================================================================
# 4. Protobuf serialization (unit, no DB)
# ============================================================================


class TestProtobufUpdateMode:
    """QueueMessage protobuf carries block_filter_tags_update_mode."""

    def test_field_round_trips(self):
        from mirix.queue.message_pb2 import QueueMessage

        msg = QueueMessage()
        msg.block_filter_tags_update_mode = "replace"
        assert msg.block_filter_tags_update_mode == "replace"

    def test_field_defaults_to_empty(self):
        from mirix.queue.message_pb2 import QueueMessage

        msg = QueueMessage()
        assert msg.block_filter_tags_update_mode == ""

    def test_has_field_detection(self):
        from mirix.queue.message_pb2 import QueueMessage

        msg = QueueMessage()
        assert not msg.HasField("block_filter_tags_update_mode")
        msg.block_filter_tags_update_mode = "merge"
        assert msg.HasField("block_filter_tags_update_mode")

    def test_serialization_round_trip(self):
        from mirix.queue.message_pb2 import QueueMessage

        msg = QueueMessage()
        msg.agent_id = "agent-test"
        msg.block_filter_tags_update_mode = "replace"
        data = msg.SerializeToString()

        msg2 = QueueMessage()
        msg2.ParseFromString(data)
        assert msg2.block_filter_tags_update_mode == "replace"
        assert msg2.agent_id == "agent-test"


# ============================================================================
# 5. Queue end-to-end Tests (integration, requires DB for org lookup)
# ============================================================================


@pytest.mark.integration
class TestQueueUpdateMode:
    """block_filter_tags_update_mode is serialized/deserialized through protobuf."""

    def test_put_messages_serializes_update_mode(
        self, queue_clean_manager, queue_sample_client, queue_sample_messages
    ):
        """put_messages with update_mode='replace' stores it in the protobuf message."""
        manager = queue_clean_manager
        manager.initialize()

        put_messages(
            actor=queue_sample_client,
            agent_id="agent-update-mode",
            input_messages=queue_sample_messages,
            block_filter_tags={"env": "staging"},
            block_filter_tags_update_mode="replace",
        )

        msg = manager._queue.get(timeout=1.0)
        assert msg.agent_id == "agent-update-mode"
        assert msg.block_filter_tags_update_mode == "replace"

        manager.cleanup()

    def test_put_messages_default_merge(
        self, queue_clean_manager, queue_sample_client, queue_sample_messages
    ):
        """put_messages without explicit update_mode defaults to 'merge'."""
        manager = queue_clean_manager
        manager.initialize()

        put_messages(
            actor=queue_sample_client,
            agent_id="agent-default-mode",
            input_messages=queue_sample_messages,
            block_filter_tags={"env": "prod"},
        )

        msg = manager._queue.get(timeout=1.0)
        assert msg.block_filter_tags_update_mode == "merge"

        manager.cleanup()

    def test_update_mode_passed_through_to_send_messages(
        self, queue_clean_manager, queue_mock_server, queue_sample_client, queue_sample_messages
    ):
        """End-to-end: put_messages -> worker -> server.send_messages receives update_mode."""
        from mirix.queue import initialize_queue

        manager = queue_clean_manager
        initialize_queue(queue_mock_server)

        put_messages(
            actor=queue_sample_client,
            agent_id="agent-e2e-mode",
            input_messages=queue_sample_messages,
            block_filter_tags={"env": "staging"},
            block_filter_tags_update_mode="replace",
        )

        time.sleep(1.5)

        assert queue_mock_server.send_messages.call_count >= 1
        call_args = queue_mock_server.send_messages.call_args
        assert call_args.kwargs.get("block_filter_tags_update_mode") == "replace"

        manager.cleanup()


# ============================================================================
# 6. BlockManager.update_block_filter_tags Tests (integration, requires DB)
# ============================================================================


@pytest.mark.integration
class TestBlockManagerUpdateFilterTags:
    """Integration tests for BlockManager.update_block_filter_tags."""

    @pytest.fixture
    def setup(self):
        from mirix.schemas.block import Block as PydanticBlock
        from mirix.schemas.client import Client as PydanticClient
        from mirix.schemas.organization import Organization as PydanticOrganization
        from mirix.schemas.user import User as PydanticUser
        from mirix.services.block_manager import BlockManager
        from mirix.services.client_manager import ClientManager
        from mirix.services.organization_manager import OrganizationManager
        from mirix.services.user_manager import UserManager

        org_id = _test_id("bft-org")
        org_mgr = OrganizationManager()
        try:
            org = org_mgr.get_organization_by_id(org_id)
        except Exception:
            org = org_mgr.create_organization(PydanticOrganization(id=org_id, name="BFT Test Org"))

        user_mgr = UserManager()
        user_id = _test_id("bft-user")
        try:
            user = user_mgr.get_user_by_id(user_id)
        except Exception:
            user = user_mgr.create_user(
                PydanticUser(id=user_id, name="BFT User", organization_id=org.id, timezone="UTC")
            )

        client_mgr = ClientManager()
        client_id = _test_id("bft-client")
        client = client_mgr.create_client(
            PydanticClient(id=client_id, organization_id=org.id, write_scope="bft-scope")
        )

        bm = BlockManager()
        block = bm.create_or_update_block(
            PydanticBlock(
                id=PydanticBlock._generate_id(),
                label="human",
                value="test",
                user_id=user.id,
                organization_id=org.id,
                filter_tags={"scope": "bft-scope", "env": "prod"},
            ),
            actor=client,
        )

        return SimpleNamespace(bm=bm, block=block, client=client, user=user)

    def test_update_filter_tags_persists(self, setup):
        """update_block_filter_tags writes new tags to DB."""
        new_tags = {"scope": "bft-scope", "env": "staging", "team": "platform"}
        setup.bm.update_block_filter_tags(
            block_id=setup.block.id,
            new_filter_tags=new_tags,
            actor=setup.client,
        )

        reloaded = setup.bm.get_block_by_id(setup.block.id)
        assert reloaded.filter_tags == new_tags

    def test_update_filter_tags_replaces_completely(self, setup):
        """filter_tags is fully replaced, not merged at the DB level."""
        new_tags = {"scope": "bft-scope", "new_key": "new_val"}
        setup.bm.update_block_filter_tags(
            block_id=setup.block.id,
            new_filter_tags=new_tags,
            actor=setup.client,
        )

        reloaded = setup.bm.get_block_by_id(setup.block.id)
        assert "env" not in reloaded.filter_tags
        assert reloaded.filter_tags["new_key"] == "new_val"
