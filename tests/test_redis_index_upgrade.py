"""DB-free tests for procedural memory's Redis index.

Procedural memory no longer uses Redis at all: skills are read and written in
Postgres only (a cached copy could be served after an edit or a delete, and a
Redis search bypassed the ranking and the retrieval gate). What remains is the
cleanup of what older versions left behind: at startup, an existing
``idx:procedural_memory`` index is dropped TOGETHER with its ``procedural:*``
documents, and no procedural index is ever created. The generic FT.INFO
attribute helper is still pinned. Mocked client, no Redis needed.
"""

import pytest

from mirix.database.redis_client import RedisMemoryClient


class _FakeFT:
    def __init__(self, info_result=None, info_raises=False, drop_raises=False):
        self._info_result = info_result
        self._info_raises = info_raises
        self._drop_raises = drop_raises
        self.drop_calls = []
        self.create_calls = []

    async def info(self):
        if self._info_raises:
            raise Exception("Unknown index name")
        return self._info_result

    async def dropindex(self, delete_documents=False):
        if self._drop_raises:
            raise Exception("cannot drop")
        self.drop_calls.append({"delete_documents": delete_documents})

    async def create_index(self, schema, definition=None):
        self.create_calls.append({"schema": schema, "definition": definition})


class _FakeRedis:
    def __init__(self, ft):
        self._ft = ft

    def ft(self, index_name):
        return self._ft


def _client_with(ft) -> RedisMemoryClient:
    client = RedisMemoryClient.__new__(RedisMemoryClient)
    client.client = _FakeRedis(ft)
    return client


def _skill_schema_info():
    return {"attributes": [[b"identifier", b"$.retrieval_embedding",
                            b"attribute", b"retrieval_embedding", b"type", b"VECTOR"]]}


def _two_vector_skill_info():
    """The skill schema before the retrieval-field migration."""
    return {"attributes": [
        [b"identifier", b"$.description_embedding", b"attribute", b"description_embedding", b"type", b"VECTOR"],
        [b"identifier", b"$.instructions_embedding", b"attribute", b"instructions_embedding", b"type", b"VECTOR"],
    ]}


def _pre_skill_info():
    return {"attributes": [
        [b"identifier", b"$.summary", b"attribute", b"summary", b"type", b"TEXT"],
        [b"identifier", b"$.summary_embedding", b"attribute", b"summary_embedding", b"type", b"VECTOR"],
        [b"identifier", b"$.steps_embedding", b"attribute", b"steps_embedding", b"type", b"VECTOR"],
    ]}


class TestIndexHasAttribute:
    def test_finds_attribute_in_bytes_tokens(self):
        assert RedisMemoryClient._index_has_attribute(
            _skill_schema_info(), "retrieval_embedding"
        )

    def test_missing_attribute(self):
        assert not RedisMemoryClient._index_has_attribute(
            _pre_skill_info(), "retrieval_embedding"
        )
        assert not RedisMemoryClient._index_has_attribute(
            _two_vector_skill_info(), "retrieval_embedding"
        )

    def test_str_tokens_and_str_key(self):
        info = {"attributes": [["attribute", "retrieval_embedding"]]}
        assert RedisMemoryClient._index_has_attribute(info, "retrieval_embedding")

    def test_bytes_top_level_key(self):
        info = {b"attributes": [[b"attribute", b"retrieval_embedding"]]}
        assert RedisMemoryClient._index_has_attribute(info, "retrieval_embedding")

    def test_non_dict_info_is_false(self):
        assert not RedisMemoryClient._index_has_attribute(None, "x")
        assert not RedisMemoryClient._index_has_attribute([], "x")


@pytest.mark.asyncio
class TestLegacyProceduralIndexIsDropped:
    async def test_no_legacy_index_is_a_no_op(self):
        ft = _FakeFT(info_raises=True)
        await _client_with(ft)._drop_legacy_procedural_index()

        assert ft.drop_calls == []
        assert ft.create_calls == []

    @pytest.mark.parametrize(
        "info", [_skill_schema_info(), _two_vector_skill_info(), _pre_skill_info()]
    )
    async def test_any_legacy_index_is_dropped_with_its_documents(self, info):
        ft = _FakeFT(info_result=info)
        await _client_with(ft)._drop_legacy_procedural_index()

        # The stale skill documents go too: nothing may serve them any more.
        assert ft.drop_calls == [{"delete_documents": True}]
        assert ft.create_calls == []

    async def test_a_drop_failure_is_logged_not_raised(self):
        ft = _FakeFT(info_result=_skill_schema_info(), drop_raises=True)
        await _client_with(ft)._drop_legacy_procedural_index()

        assert ft.create_calls == []

    async def test_create_indexes_never_creates_a_procedural_index(self, monkeypatch):
        client = RedisMemoryClient.__new__(RedisMemoryClient)
        calls = []

        async def record(name):
            calls.append(name)

        for method in (
            "_create_block_index",
            "_create_message_index",
            "_create_organization_index",
            "_create_user_index",
            "_create_agent_index",
            "_create_tool_index",
            "_create_episodic_index",
            "_create_semantic_index",
            "_create_resource_index",
            "_create_knowledge_index",
            "_drop_legacy_procedural_index",
        ):
            monkeypatch.setattr(
                client, method, lambda method=method: record(method), raising=True
            )
        await client.create_indexes()

        assert "_drop_legacy_procedural_index" in calls
        assert not hasattr(RedisMemoryClient, "_create_procedural_index")
