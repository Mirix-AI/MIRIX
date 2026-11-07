"""embed_batch gives the graph the provider's vectors, without main's zero padding.

embedding_model() returns MAX_EMBEDDING_DIM-wide vectors (padded for pgvector); the
Neo4j vector indexes hold settings.neo4j_vector_dim dimensions, so a padded vector
would fail every db.index.vector.queryNodes / setNodeVectorProperty call.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_embed_batch_cuts_the_padding(monkeypatch):
    from mirix.constants import MAX_EMBEDDING_DIM
    from mirix.services import _graph_common

    class PaddingModel:
        async def get_text_embedding(self, text):
            return [0.5] * 1536 + [0.0] * (MAX_EMBEDDING_DIM - 1536)

    async def fake_embedding_model(config):
        return PaddingModel()

    monkeypatch.setattr(_graph_common, "embedding_model", fake_embedding_model)
    agent_state = SimpleNamespace(embedding_config=SimpleNamespace(embedding_dim=1536))
    out = await _graph_common.embed_batch(["Caroline", "support group"], agent_state)
    assert [len(v) for v in out] == [1536, 1536]
    assert out[0] == [0.5] * 1536


@pytest.mark.asyncio
async def test_embed_batch_keeps_a_provider_sized_vector(monkeypatch):
    from mirix.services import _graph_common

    class RawModel:
        async def get_text_embedding(self, text):
            return [0.25] * 1536

    async def fake_embedding_model(config):
        return RawModel()

    monkeypatch.setattr(_graph_common, "embedding_model", fake_embedding_model)
    agent_state = SimpleNamespace(embedding_config=SimpleNamespace(embedding_dim=1536))
    assert await _graph_common.embed_batch(["Melanie"], agent_state) == [[0.25] * 1536]
