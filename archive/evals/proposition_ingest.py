"""v7.3 ingest: build LongMemEval-S / MAB memory from ATOMIC PROPOSITIONS.

ONE LLM call per chunk emits both the propositions and their typed entities. Each
proposition becomes a fine-grained semantic_memory (embedded); its entities are
handed straight to the V7 graph build, which consumes only name + entity_type.
LightRAG is never invoked -- it was ~97% of graph-build time, and its relations
and descriptions were discarded by v7 anyway.

Env:
  MIRIX_PG_DB           target DB (schema must exist -- start the server against it once)
  MIRIX_ENABLE_GRAPH_MEMORY=true, MIRIX_GRAPH_VERSION=v7.3
  SOURCE                HF metadata.source (default longmemeval_s*)
  MAX_CHUNKS            optional cap for a smoke test
Usage:  python evals/mab/proposition_ingest.py
"""
import asyncio
import datetime
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(__file__))
from sqlalchemy import text as sa_text

import longmem_eval as L
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.database.neo4j_client import get_neo4j_driver, init_neo4j_client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.server.server import db_context
from mirix.services._graph_common import embed_batch
from mirix.services.graph_memory_manager_v7 import V7GraphManager
from mirix.services.lightrag_extractor import ExtractedEntity
from mirix.services.proposition_extractor import extract_propositions_with_entities
from mirix.settings import settings

EMB = EmbeddingConfig(
    embedding_endpoint_type="openai", embedding_endpoint="https://api.openai.com/v1",
    embedding_model="text-embedding-3-small", embedding_dim=1536, embedding_chunk_size=300)
LLM = LLMConfig(model="gpt-4.1-mini", model_endpoint_type="openai",
    model_endpoint="https://api.openai.com/v1", context_window=128000)
AST = types.SimpleNamespace(embedding_config=EMB, llm_config=LLM)
UID, ORG = "longmem_s_0", "mirix-eval-org"
SOURCE = os.environ.get("SOURCE", "longmemeval_s*")
MAXC = int(os.environ.get("MAX_CHUNKS", "0")) or None
API_KEY = os.environ["OPENAI_API_KEY"]

INSERT = sa_text(
    "INSERT INTO semantic_memory (id,name,summary,details,source,source_refs,prior_values,"
    "last_modify,created_at,is_deleted,user_id,organization_id,summary_embedding) VALUES "
    "(:id,:nm,:sm,:dt,:src,'[]','[]','{}',:ts,false,:u,:o,CAST(:emb AS vector))")


async def main():
    await init_neo4j_client()
    drv = get_neo4j_driver()
    async with drv.session(database=settings.neo4j_database) as s:
        # consume() so the delete actually completes before we start writing
        await (await s.run("MATCH (n) DETACH DELETE n")).consume()
        left = (await (await s.run("MATCH (n) RETURN count(n) AS c")).single())["c"]
    print(f"neo4j wiped (remaining nodes: {left})", flush=True)

    it = L.load_longmem_s(source=SOURCE, limit=1)[0]
    chunks = L.parse_sessions(it["context"])
    if MAXC:
        chunks = chunks[:MAXC]
    print(f"chunks={len(chunks)} source={SOURCE}", flush=True)

    mgr = V7GraphManager()
    total = n_ents = 0
    t0 = time.time()
    for ci, ch in enumerate(chunks, 1):
        props = await extract_propositions_with_entities(ch["text"], api_key=API_KEY)
        if not props:
            print(f"chunk {ci}/{len(chunks)}: no propositions", flush=True)
            continue

        embs = await embed_batch([p.text for p in props], AST)
        rows = []  # (memory_id, Proposition, embedding)
        for p, e in zip(props, embs):
            if e is None:
                continue
            total += 1
            rows.append((f"prop_{total:06d}", p, e))

        async with db_context() as session:
            for mid, p, e in rows:
                epad = str(list(e) + [0.0] * (MAX_EMBEDDING_DIM - len(e)))
                await session.execute(INSERT, {
                    "id": mid, "nm": p.text[:120], "sm": p.text, "dt": p.text,
                    "src": f"chunk_{ci - 1}", "ts": datetime.datetime.utcnow(),
                    "u": UID, "o": ORG, "emb": epad})
            await session.commit()

        for mid, p, _ in rows:
            n_ents += len(p.entities)
            try:
                await mgr.process_memory(
                    source_kind="semantic", source_id=mid, text=p.text, title=p.text[:80],
                    summary=p.text, source_meta=None, agent_state=AST,
                    organization_id=ORG, user_id=UID,
                    entities=[ExtractedEntity(name=x.name, entity_type=x.type, description="")
                              for x in p.entities])
            except Exception as ex:  # noqa: BLE001
                print(f"  graph fail {mid}: {str(ex)[:80]}", flush=True)

        print(f"chunk {ci}/{len(chunks)}  props={total} entities={n_ents} "
              f"(+{time.time() - t0:.0f}s)", flush=True)

    print(f"DONE: {total} propositions, {n_ents} entity mentions, "
          f"{time.time() - t0:.0f}s, {len(chunks)} LLM extraction calls", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
