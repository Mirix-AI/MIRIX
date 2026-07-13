"""DB-free tests for the procedural Redis index schema upgrade.

An FT index created before the skill schema (summary / summary_embedding /
steps_embedding) persists across deploys. Because index creation used to
early-return whenever FT.INFO succeeded, that stale index was never rebuilt and
every skill-field query (description/instructions, description_embedding KNN)
missed Redis forever. These tests pin the upgrade behavior with a mocked
client: fresh → create; current schema → leave alone; stale schema → drop the
index (keep documents) and recreate.
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
    return {"attributes": [[b"identifier", b"$.description_embedding",
                            b"attribute", b"description_embedding", b"type", b"VECTOR"]]}


def _pre_skill_info():
    return {"attributes": [
        [b"identifier", b"$.summary", b"attribute", b"summary", b"type", b"TEXT"],
        [b"identifier", b"$.summary_embedding", b"attribute", b"summary_embedding", b"type", b"VECTOR"],
        [b"identifier", b"$.steps_embedding", b"attribute", b"steps_embedding", b"type", b"VECTOR"],
    ]}


class TestIndexHasAttribute:
    def test_finds_attribute_in_bytes_tokens(self):
        assert RedisMemoryClient._index_has_attribute(
            _skill_schema_info(), "description_embedding"
        )

    def test_missing_attribute(self):
        assert not RedisMemoryClient._index_has_attribute(
            _pre_skill_info(), "description_embedding"
        )

    def test_str_tokens_and_str_key(self):
        info = {"attributes": [["attribute", "description_embedding"]]}
        assert RedisMemoryClient._index_has_attribute(info, "description_embedding")

    def test_bytes_top_level_key(self):
        info = {b"attributes": [[b"attribute", b"description_embedding"]]}
        assert RedisMemoryClient._index_has_attribute(info, "description_embedding")

    def test_non_dict_info_is_false(self):
        assert not RedisMemoryClient._index_has_attribute(None, "x")
        assert not RedisMemoryClient._index_has_attribute([], "x")


@pytest.mark.asyncio
class TestCreateProceduralIndexUpgrade:
    async def test_fresh_index_is_created(self):
        ft = _FakeFT(info_raises=True)
        await _client_with(ft)._create_procedural_index()

        assert len(ft.create_calls) == 1
        assert ft.drop_calls == []

    async def test_current_schema_left_alone(self):
        ft = _FakeFT(info_result=_skill_schema_info())
        await _client_with(ft)._create_procedural_index()

        assert ft.create_calls == []
        assert ft.drop_calls == []

    async def test_stale_pre_skill_index_dropped_and_recreated(self):
        ft = _FakeFT(info_result=_pre_skill_info())
        await _client_with(ft)._create_procedural_index()

        # The documents must be preserved: index-only drop.
        assert ft.drop_calls == [{"delete_documents": False}]
        assert len(ft.create_calls) == 1

    async def test_drop_failure_does_not_recreate_over_live_index(self):
        ft = _FakeFT(info_result=_pre_skill_info(), drop_raises=True)
        await _client_with(ft)._create_procedural_index()

        assert ft.create_calls == []
