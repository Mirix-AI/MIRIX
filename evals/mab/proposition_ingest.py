"""v7.3 ingest: build LongMemEval-S / MAB memory from ATOMIC PROPOSITIONS.

Per chunk: one proposition-extraction call (mirix.services.proposition_extractor)
-> each proposition becomes a fine-grained semantic_memory (embedded) + is indexed
by the existing V7 graph build (anchor -> DESCRIBED_BY -> ConceptRef).

Env:
  MIRIX_PG_DB           target DB (schema must exist — start the server against it once)
  MIRIX_ENABLE_GRAPH_MEMORY=true, MIRIX_GRAPH_VERSION=v7.3
  SOURCE                HF metadata.source (default longmemeval_s*)
  MAX_CHUNKS            optional cap for a smoke test
Usage:  python evals/mab/proposition_ingest.py
"""
import asyncio
import datetime
import os
import sys
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
from mirix.services.proposition_extractor import extract_propositions
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
        await s.run("MATCH (n) DETACH DELETE n")
    print("neo4j wiped", flush=True)

    it = L.load_longmem_s(source=SOURCE, limit=1)[0]
    chunks = L.parse_sessions(it["context"])
    if MAXC:
        chunks = chunks[:MAXC]
    print(f"chunks={len(chunks)} source={SOURCE}", flush=True)

    mgr = V7GraphManager()
    total = 0
    for ci, ch in enumerate(chunks):
        props = await extract_propositions(ch["text"], api_key=API_KEY)
        if not props:
            continue
        embs = await embed_batch(props, AST)
        rows = []  # (mid, proposition)
        for p, e in zip(props, embs):
            if e is None:
                continue
            total += 1
            rows.append((f"prop_{total:06d}", p, e))
        async with db_context() as session:
            for mid, p, e in rows:
                epad = str(list(e) + [0.0] * (MAX_EMBEDDING_DIM - len(e)))
                await session.execute(INSERT, {
                    "id": mid, "nm": p[:120], "sm": p, "dt": p, "src": f"chunk_{ci}",
                    "ts": datetime.datetime.utcnow(), "u": UID, "o": ORG, "emb": epad})
            await session.commit()
        for mid, p, e in rows:
            try:
                await mgr.process_memory(
                    source_kind="semantic", source_id=mid, text=p, title=p[:80],
                    summary=p, source_meta=None, agent_state=AST,
                    organization_id=ORG, user_id=UID)
            except Exception as ex:  # noqa: BLE001
                print(f"  graph fail {mid}: {str(ex)[:80]}", flush=True)
        if ci % 10 == 0:
            print(f"chunk {ci}/{len(chunks)}  propositions {total}", flush=True)

    print(f"DONE: {total} propositions ingested", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
