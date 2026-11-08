"""Read-only falsification of entity consolidation. No writes, no LLM calls.

THE QUESTION. 17.5% of anchors (1,636 of 9,341 on the v7.23 full store) are surface
fragments of another anchor with the same head noun — conv-44 holds `dogs`, `her dogs`,
`two dogs`, `all dogs`, `city dogs` as five nodes for one referent. Retrieval matches
anchor NAMES, so a query hits one fragment and reaches only the PG rows attached to that
fragment; the rows hanging off its siblings are unreachable for that query. Consolidating
them would make one hit reach the union.

Whether that is worth a multi-hour re-ingest is decided here, before paying for it.

WHAT IS SIMULATED. For each question that v7.24 got wrong:
  baseline  = union of episodic_ids + semantic_ids over the anchors the REAL vector
              search returns (top-18, via V7Retriever._search_anchors)
  merged    = the same, but each anchor contributes its whole cluster's ids
  payoff    = a PG row in `merged - baseline` whose text supports the gold answer

Only the direct anchor->PG path is simulated. The frame hop is reported separately
because its 240-fact cap interacts with degree, and consolidation raises degree — that
confound is the reason ORDER BY was added to the cap first.

KILL THRESHOLD: fewer than 15 of the wrong questions gaining a gold-bearing row. The
measurement noise on this benchmark is +/-6-7 questions per arm at one run each, so a
ceiling below 15 cannot be demonstrated even if every gained row converted.

    python consolidation_audit.py --limit 0        # all wrong questions
"""
import argparse
import asyncio
import collections
import json
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RESULTS = ("/home/lj/code/MIRIX/evals/results/locomo/"
           "v724_full_qaonly_v723graph_r1/metrics.json")
PREFIX = "v723full-r1__"
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did what when where who how why which their her "
           "his its she he they them about into over under after before not".split())


def content(text) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in STOP}


def head(name: str) -> str:
    toks = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split()
    return toks[-1] if toks else ""


# Non-restrictive modifiers: they mark a referent already established in context but do
# not narrow WHICH one it is, so dropping them cannot merge two distinct referents.
# Everything else — adjectives, numerals, ordinals, proper nouns — is RESTRICTIVE and
# exists precisely to distinguish. This is a closed, language-general class, not a
# corpus-derived list.
_NON_RESTRICTIVE = frozenset("""
a an the this that these those my your his her its our their
""".split())


def cluster(names_by_id: dict) -> dict:
    """anchor id -> cluster id.

    Merge two anchors only when they share a head noun AND the tokens by which they
    differ are all NON-RESTRICTIVE. The first version of this audit merged on bare
    token-subset containment and produced, on conv-42:

        tournament | first video game tournament | second gaming tournament
                   | fourth video game tournament
        screenplay | third screenplay | first full screenplay
        turtles    | two turtles | three turtles

    — collapsing exactly the ordinal and cardinal distinctions that the failing
    questions turn on ("her THIRD screenplay", "his FIRST tournament", "how many
    turtles"). It scored a large reachability gain by dumping a topic's whole
    neighbourhood into one node, which is recall bought by destroying the answer.
    Requiring the difference to be non-restrictive is what separates "same referent,
    two surface forms" from "two referents, one hypernym".
    """
    toks = {i: set(re.sub(r"[^a-z0-9 ]", " ", n.lower()).split())
            for i, n in names_by_id.items()}
    by_head = collections.defaultdict(list)
    for i, n in names_by_id.items():
        h = head(n)
        if len(h) >= 4:
            by_head[h].append(i)
    parent = {i: i for i in names_by_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for h, ids in by_head.items():
        if len(ids) < 2:
            continue
        for a in ids:
            for b in ids:
                if a is b:
                    continue
                # proper subset AND every extra token non-restrictive:
                #   "her dogs" / "the dogs"  -> same referent, merge
                #   "dogs" / "two dogs"      -> the numeral restricts, do NOT merge
                if toks[a] and toks[a] < toks[b] and (toks[b] - toks[a]) <= _NON_RESTRICTIVE:
                    union(b, a)
    return {i: find(i) for i in names_by_id}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all wrong questions")
    ap.add_argument("--top-k", type=int, default=18)
    a = ap.parse_args()

    import types
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.database.neo4j_client import init_neo4j_client, get_neo4j_driver
    from mirix.services.graph_retriever_v7 import V7Retriever
    from mirix.services._graph_common import embed_batch
    from mirix.settings import settings
    from sqlalchemy import text as sa_text
    from mirix.server.server import db_context

    ag = types.SimpleNamespace(
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536, embedding_chunk_size=300),
        llm_config=LLMConfig(model="gpt-4.1-mini", model_endpoint_type="openai",
                             model_endpoint="https://api.openai.com/v1",
                             context_window=128000))

    wrong = [r for r in json.load(open(RESULTS))["llm_judge_results"]
             if r.get("score") == 0]
    if a.limit:
        wrong = wrong[: a.limit]
    print(f"{len(wrong)} wrong questions", flush=True)

    await init_neo4j_client()
    driver = get_neo4j_driver()
    retr = V7Retriever()

    # ---- anchors and their PG ids, per conversation ----
    anchors: dict = {}
    async with driver.session(database=settings.neo4j_database) as s:
        res = await s.run(
            "MATCH (x:V7Anchor) WHERE x.user_id STARTS WITH $p "
            "RETURN x.user_id AS u, x.id AS id, x.name AS name, "
            "       coalesce(x.episodic_ids,[]) AS ep, coalesce(x.semantic_ids,[]) AS sem",
            p=PREFIX)
        async for rec in res:
            anchors.setdefault(rec["u"], {})[rec["id"]] = (
                rec["name"], set(rec["ep"]) | set(rec["sem"]))

    clusters, cl_ids = {}, {}
    for u, d in anchors.items():
        cmap = cluster({i: v[0] for i, v in d.items()})
        clusters[u] = cmap
        agg = collections.defaultdict(set)
        for i, root in cmap.items():
            agg[root] |= d[i][1]
        cl_ids[u] = agg
    merged_n = sum(1 for u in clusters for i, r in clusters[u].items() if i != r)
    print(f"structural clusters: {merged_n} anchors absorbed into a longer sibling "
          f"({merged_n / sum(len(d) for d in anchors.values()):.1%} of all anchors)",
          flush=True)

    # ---- PG text per memory id ----
    texts: dict = {}
    async with db_context() as session:
        for table, cols in (("episodic_memory", "summary, details"),
                            ("semantic_memory", "name, summary, details")):
            rows = (await session.execute(sa_text(
                f"SELECT id, {cols} FROM {table} WHERE NOT is_deleted"))).fetchall()
            for row in rows:
                texts[row[0]] = " ".join(str(x or "") for x in row[1:])
    print(f"{len(texts)} PG rows loaded", flush=True)

    stat = collections.Counter()
    gained_examples = []
    sem_lock = asyncio.Semaphore(6)

    async def one(r):
        u = PREFIX + r["sample_id"]
        if u not in anchors:
            return
        async with sem_lock:
            emb = (await embed_batch([r["question"]], ag))[0]
            hits = await retr._search_anchors(driver, u, emb, a.top_k)
        base, merged = set(), set()
        for h in hits:
            if h.id not in anchors[u]:
                continue
            base |= anchors[u][h.id][1]
            merged |= cl_ids[u][clusters[u][h.id]]
        delta = merged - base
        stat["questions"] += 1
        stat["extra_rows"] += len(delta)
        gold = content(r["expected_answer"])
        if not gold:
            return
        if any(len(gold & content(texts.get(mid, ""))) / len(gold) >= 0.6 for mid in delta):
            stat["gained_gold_row"] += 1
            gained_examples.append(r)
        if any(len(gold & content(texts.get(mid, ""))) / len(gold) >= 0.6 for mid in base):
            stat["gold_already_reachable"] += 1

    await asyncio.gather(*[one(r) for r in wrong])

    n = stat["questions"]
    print(f"\n=== consolidation audit over {n} wrong questions ===")
    print(f"  gold-bearing row ALREADY reachable   {stat['gold_already_reachable']:4d}"
          f"  ({stat['gold_already_reachable']/n:.1%})   <- merging cannot help these")
    print(f"  gold-bearing row NEWLY reachable     {stat['gained_gold_row']:4d}"
          f"  ({stat['gained_gold_row']/n:.1%})   <- the entire ceiling of this idea")
    print(f"  extra PG rows opened up, total       {stat['extra_rows']:4d}"
          f"  ({stat['extra_rows']/n:.1f} per question)")
    print(f"\n  KILL THRESHOLD 15 -> "
          f"{'PASS, worth a real ingest' if stat['gained_gold_row'] >= 15 else 'KILL: ceiling is below the noise floor'}")
    for r in gained_examples[:8]:
        print(f"    [{r['sample_id']}] {r['question'][:62]}  gold={str(r['expected_answer'])[:34]}")


asyncio.run(main())
