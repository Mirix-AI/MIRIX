"""
Tests for the constant-field normalizer registry (tool_normalizers.py).

Verifies that prompt-mandated constant fields (actor, event_type, source)
are defaulted when the LLM omits them from its tool-call args, before the
tool function ever runs (ECMS-534). These fast unit tests call
normalize_tool_args directly — no manager mocks or running server needed.
"""

import pytest

from mirix.agent.tool_normalizers import normalize_tool_args

pytestmark = pytest.mark.asyncio(loop_scope="module")


class TestEpisodicMemoryInsertMissingActor:
    async def test_defaults_actor_to_user_when_omitted(self):
        args = {"items": [{"event_type": "activity", "summary": "s", "details": "d"}]}

        normalize_tool_args("episodic_memory_insert", args)

        assert args["items"][0]["actor"] == "user"


class TestEpisodicMemoryInsertMissingEventType:
    async def test_defaults_event_type_to_user_message_when_omitted(self):
        args = {"items": [{"actor": "user", "summary": "s", "details": "d"}]}

        normalize_tool_args("episodic_memory_insert", args)

        assert args["items"][0]["event_type"] == "user_message"


class TestEpisodicMemoryInsertPreservesExplicitValues:
    async def test_does_not_override_present_fields(self):
        args = {
            "items": [
                {"actor": "assistant", "event_type": "custom", "summary": "s", "details": "d"}
            ]
        }

        normalize_tool_args("episodic_memory_insert", args)

        assert args["items"][0]["actor"] == "assistant"
        assert args["items"][0]["event_type"] == "custom"


class TestEpisodicMemoryReplaceMissingActor:
    async def test_defaults_actor_to_user_when_omitted(self):
        args = {"new_items": [{"event_type": "activity", "summary": "s", "details": "d"}]}

        normalize_tool_args("episodic_memory_replace", args)

        assert args["new_items"][0]["actor"] == "user"


class TestEpisodicMemoryReplaceMissingEventType:
    async def test_defaults_event_type_to_user_message_when_omitted(self):
        args = {"new_items": [{"actor": "user", "summary": "s", "details": "d"}]}

        normalize_tool_args("episodic_memory_replace", args)

        assert args["new_items"][0]["event_type"] == "user_message"


class TestSemanticMemoryInsertMissingSource:
    async def test_defaults_source_to_user_message_when_omitted(self):
        args = {"items": [{"name": "n", "summary": "s", "details": "d"}]}

        normalize_tool_args("semantic_memory_insert", args)

        assert args["items"][0]["source"] == "user message"


class TestSemanticMemoryUpdateMissingSource:
    async def test_defaults_source_to_user_message_when_omitted(self):
        args = {"new_items": [{"name": "n", "summary": "s", "details": "d"}]}

        normalize_tool_args("semantic_memory_update", args)

        assert args["new_items"][0]["source"] == "user message"


class TestKnowledgeVaultInsertMissingSource:
    async def test_defaults_source_to_user_message_when_omitted(self):
        args = {
            "items": [
                {"entry_type": "secret", "sensitivity": "high", "secret_value": "val", "caption": "cap"}
            ]
        }

        normalize_tool_args("knowledge_vault_insert", args)

        assert args["items"][0]["source"] == "user message"


class TestKnowledgeVaultUpdateMissingSource:
    async def test_defaults_source_to_user_message_when_omitted(self):
        args = {
            "new_items": [
                {"entry_type": "secret", "sensitivity": "high", "secret_value": "val", "caption": "cap"}
            ]
        }

        normalize_tool_args("knowledge_vault_update", args)

        assert args["new_items"][0]["source"] == "user message"


class TestNormalizeToolArgsNoOp:
    async def test_unregistered_tool_is_a_no_op(self):
        args = {"items": [{"summary": "s"}]}

        normalize_tool_args("procedural_memory_insert", args)

        assert args == {"items": [{"summary": "s"}]}

    async def test_missing_items_key_is_a_no_op(self):
        args = {}

        normalize_tool_args("episodic_memory_insert", args)

        assert args == {}

    async def test_empty_items_list_is_a_no_op(self):
        args = {"items": []}

        normalize_tool_args("episodic_memory_insert", args)

        assert args == {"items": []}
