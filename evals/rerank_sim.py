"""Would ranking on a better signal move the supporting row up? Read-only simulation.

rank_audit.py established that bucket C is substantially a ranking failure: when the
answer is correct the supporting row is first 69.7% of the time, when it is wrong only
30.1%. The final ordering in graph_retriever_v7 is

    ORDER BY summary_embedding <=> query    LIMIT k

— a single cosine over the SUMMARY field, which averages 61-94 characters, while the
same rows carry a fully-populated details_embedding over a 290-character field that
ranking never consults.

This re-orders the SAME candidate rows under alternative scores and reports where the
supporting row lands. No code change, no ingest, no LLM: the embeddings already exist,
so this is arithmetic over what is already stored.

Variants compared:
    summary          what ships today
    details          details_embedding alone
    max              max of the two cosines
    mean             their average

If none of them lifts the supporting row for the wrong answers, ranking on text
similarity is exhausted and the lever has to be a non-textual signal — which is the
graph's structural score, currently computed for facts and discarded before rows are
ordered.

    python rerank_sim.py
"""
import asyncio
import collections
import json
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RESULTS = ("/home/lj/code/MIRIX/evals/results/locomo/"
           "v724_full_qaonly_v723graph_r1/metrics.json")
PREFIX = "v723full-r1__"
SUPPORT = 0.6
TOPK = 15
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did what when where who how why which their her "
           "his its she he they them about into over under after before not".split())


def content(text) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in STOP}


async def main() -> None:
    import types
    import numpy as np
    from sqlalchemy import text as sa_text
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.server.server import db_context
    from mirix.services._graph_common import embed_batch

    ag = types.SimpleNamespace(
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536, embedding_chunk_size=300),
        llm_config=LLMConfig(model="gpt-4.1-mini", model_endpoint_type="openai",
                             model_endpoint="https://api.openai.com/v1",
                             context_window=128000))

    judged = json.load(open(RESULTS))["llm_judge_results"]
    wrong = [r for r in judged if r.get("score") == 0]

    # candidate pool per conversation: every row, with both embeddings
    pool: dict = {}
    async with db_context() as session:
        for table, cols in (("episodic_memory", "summary, details"),
                            ("semantic_memory", "summary, details")):
            rows = (await session.execute(sa_text(
                f"SELECT user_id, id, {cols}, summary_embedding, details_embedding "
                f"FROM {table} WHERE NOT is_deleted "
                f"  AND summary_embedding IS NOT NULL"))).fetchall()
            def vec(v):
                # pgvector comes back as its text form over this driver.
                if v is None:
                    return None
                if isinstance(v, str):
                    return np.fromstring(v.strip()[1:-1], sep=",", dtype=np.float32)
                return np.asarray(v, dtype=np.float32)

            for u, mid, summ, det, se, de in rows:
                sv = vec(se)
                if sv is None or not sv.size:
                    continue
                pool.setdefault(u, []).append(
                    (mid, f"{summ or ''} {det or ''}", sv, vec(de)))
    for u in pool:
        ids, txt, se, de = zip(*pool[u])
        S = np.stack(se)
        D = np.stack([d if d is not None and d.size == s.size else s
                      for d, s in zip(de, se)])
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        pool[u] = (list(ids), list(txt), S, D)
    print(f"pool built for {len(pool)} conversations", flush=True)

    variants = ("summary", "details", "max", "mean")
    hits = {v: collections.Counter() for v in variants}
    found = {v: 0 for v in variants}
    n = 0
    sem = asyncio.Semaphore(8)

    async def one(r):
        nonlocal n
        u = PREFIX + r["sample_id"]
        if u not in pool:
            return
        gold = content(r["expected_answer"])
        if not gold:
            return
        async with sem:
            q = (await embed_batch([r["question"]], ag))[0]
        import numpy as _np
        qv = _np.asarray(q, dtype=_np.float32)
        qv /= _np.linalg.norm(qv) + 1e-9
        ids, txt, S, D = pool[u]
        # Stored vectors are zero-padded to MAX_EMBEDDING_DIM (4096); the live query is
        # the model's native 1536. The padding is zeros, so truncating to the query's
        # width is exact, not an approximation.
        d = qv.size
        cs, cd = S[:, :d] @ qv, D[:, :d] @ qv
        scores = {"summary": cs, "details": cd,
                  "max": _np.maximum(cs, cd), "mean": (cs + cd) / 2}
        n += 1
        for v in variants:
            order = _np.argsort(-scores[v])[:TOPK]
            rank = None
            for pos, idx in enumerate(order, start=1):
                if len(gold & content(txt[idx])) / len(gold) >= SUPPORT:
                    rank = pos
                    break
            if rank is None:
                continue
            found[v] += 1
            hits[v]["1" if rank == 1 else "2-3" if rank <= 3 else
                    "4-7" if rank <= 7 else "8-15"] += 1

    await asyncio.gather(*[one(r) for r in wrong])

    print(f"\n=== re-ranking the SAME pool, {n} wrong answers, top-{TOPK} ===")
    print(f"  {'variant':10s}{'support found':>15s}{'rank 1':>10s}{'top-3':>10s}")
    for v in variants:
        f = found[v]
        r1 = hits[v]["1"] / max(f, 1)
        t3 = (hits[v]["1"] + hits[v]["2-3"]) / max(f, 1)
        print(f"  {v:10s}{f:8d}/{n:<6d}{r1:10.1%}{t3:10.1%}")
    print("\n  Reference from rank_audit.py, live pipeline on these same questions:")
    print("    wrong answers   rank-1 30.1%,  top-3 53.4%")
    print("    correct answers rank-1 69.7%,  top-3 79.6%")
    print("\n  NOTE: this pool is every row in the conversation, not the graph's")
    print("  candidate set, so absolute numbers are not comparable to the live")
    print("  pipeline. The comparison that matters is BETWEEN the variants.")


asyncio.run(main())
