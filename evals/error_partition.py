"""Partition every wrong answer into exactly one bucket, so the buckets sum to the
error count and each one names a different thing to fix.

The point is coverage. Earlier audits measured specific proposals and between them
accounted for 31 of 169 errors; this asks where the other 138 are, without assuming a
mechanism first.

THE PRIMARY SPLIT is where the gold answer's content lives, checked at two scopes:

  not in the STORE at all      -> ingest never wrote it, or the gold is abstractive and
                                  no literal support exists. Retrieval is irrelevant.
  in the store, not RETRIEVED  -> retrieval miss. This is the bucket a better retriever
                                  or representation can move.
  retrieved                    -> the evidence was in front of the model and the answer
                                  is still wrong. Downstream of retrieval.

Only the third bucket is sub-classified, because only there is the failure mechanism
about answering rather than finding.

The containment test is a LOWER bound on support: abstractive gold ("Somewhat, but not
extremely religious") fails it even when the evidence is present, which inflates the
first bucket. The abstractive share is reported separately so the bound is visible
rather than hidden.

    python error_partition.py
"""
import asyncio
import collections
import json
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

# Which run to analyse, and the user_id prefix its store was written under. Defaults
# are the contaminated pre-36d8d40 run so old invocations keep reproducing; pass --run
# and --prefix for the decontaminated ones.
DEFAULT_RUN = "v724_full_qaonly_v723graph_r1"
DEFAULT_PREFIX = "v723full-r1__"
RESULTS = f"/home/lj/code/MIRIX/evals/results/locomo/{DEFAULT_RUN}/metrics.json"
PREFIX = DEFAULT_PREFIX
SUPPORT = 0.6          # share of gold content tokens that must appear

STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did what when where who how why which their her "
           "his its she he they them about into over under after before not".split())
ABSTAIN = re.compile(
    r"no (specific |explicit )?(information|mention|record|detail)|"
    r"not (explicitly |specifically )?(mentioned|stated|specified|provided|available)|"
    r"there is no|cannot (be )?determine|unable to|do(es)? not (mention|specify|indicate)",
    re.I)
HOWMANY = re.compile(r"^how (many|much|long|often)", re.I)
HEDGE = re.compile(r"\b(likely|probably|might|maybe|somewhat|possibly|suggests?)\b", re.I)


def content(text) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in STOP}


def supported(gold: set, text) -> bool:
    return bool(gold) and len(gold & content(text)) / len(gold) >= SUPPORT


def _apply_overrides(run, prefix):
    global RESULTS, PREFIX
    if run:
        RESULTS = f"/home/lj/code/MIRIX/evals/results/locomo/{run}/metrics.json"
    if prefix:
        PREFIX = prefix


def build():
    from mirix_memory_system import MirixMemorySystem
    wrong = [r for r in json.load(open(RESULTS))["llm_judge_results"]
             if r.get("score") == 0]
    convs = sorted({r["sample_id"] for r in wrong})
    clients = {c: MirixMemorySystem(
        user_id=PREFIX + c,
        mirix_config_path="/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml")
        for c in convs}
    return clients, wrong, convs


async def main(clients, wrong, convs) -> None:
    from sqlalchemy import text as sa_text
    from mirix.server.server import db_context

    # whole-store text per conversation, for the "is it anywhere?" scope
    store = collections.defaultdict(str)
    async with db_context() as session:
        for table, cols in (("episodic_memory", "summary, details"),
                            ("semantic_memory", "name, summary, details")):
            rows = (await session.execute(sa_text(
                f"SELECT user_id, {cols} FROM {table} WHERE NOT is_deleted"))).fetchall()
            for row in rows:
                store[row[0]] += " " + " ".join(str(x or "") for x in row[1:])
    store_tokens = {u: content(t) for u, t in store.items()}
    print(f"store loaded for {len(store_tokens)} users", flush=True)

    bucket = collections.Counter()
    sub = collections.Counter()
    ex = collections.defaultdict(list)
    sem = asyncio.Semaphore(6)

    async def one(r):
        u = PREFIX + r["sample_id"]
        gold = content(r["expected_answer"])
        if not gold:
            bucket["gold has no content tokens"] += 1
            return
        async with sem:
            res = await clients[r["sample_id"]].client.search(
                user_id=u, query=r["question"], memory_type="all",
                search_method="embedding", limit=15)
        rows = [x for x in (res.get("results") or []) if isinstance(x, dict)]
        blob = " ".join(f"{x.get('summary') or ''} {x.get('details') or ''} "
                        f"{x.get('name') or ''}" for x in rows)

        in_store = len(gold & store_tokens.get(u, set())) / len(gold) >= SUPPORT
        in_ret = supported(gold, blob)

        if in_ret:
            k = "C retrieved, still wrong"
            pred = str(r["predicted_answer"] or "")
            if ABSTAIN.search(pred):
                s = "C1 abstained with the evidence in hand"
            elif HOWMANY.search(r["question"]):
                s = "C2 counting/aggregation"
            elif str(r.get("category")) == "2":
                s = "C3 temporal selection"
            elif len(pred) > 200:
                s = "C4 discursive, never states the specific"
            else:
                s = "C5 stated a specific, wrong one"
            sub[s] += 1
            ex[s].append(r)
        elif in_store:
            k = "B in the store, NOT retrieved"
            ex[k].append(r)
        else:
            k = "A not in the store at all"
            # is the gold abstractive/inferential rather than a stated fact?
            s = ("A1 gold is hedged/inferential" if HEDGE.search(str(r["expected_answer"]))
                 else "A2 gold looks factual but is absent")
            sub[s] += 1
            ex[k].append(r)
        bucket[k] += 1

    await asyncio.gather(*[one(r) for r in wrong])

    total = sum(bucket.values())
    print(f"\n=== partition of {total} wrong answers ===")
    for k in sorted(bucket):
        n = bucket[k]
        print(f"  {n:4d}  ({n/total:5.1%})  {k}")
    print("\n  A sub-split (is the gold even a stated fact?):")
    for k in sorted(s for s in sub if s.startswith("A")):
        print(f"      {sub[k]:4d}  {k}")
    print("\n  C sub-split (evidence was in front of the model):")
    for k in sorted(s for s in sub if s.startswith("C")):
        print(f"      {sub[k]:4d}  {k}")

    print("\n--- B: in the store but not retrieved (the retrieval bucket) ---")
    for r in ex["B in the store, NOT retrieved"][:10]:
        print(f"  [{r['sample_id']}] {r['question'][:60]}")
        print(f"     gold: {str(r['expected_answer'])[:60]}")
    print("\n--- A2: looks factual, absent from the whole store (ingest loss) ---")
    for r in ex["A not in the store at all"][:8]:
        if not HEDGE.search(str(r["expected_answer"])):
            print(f"  [{r['sample_id']}] {r['question'][:60]}")
            print(f"     gold: {str(r['expected_answer'])[:60]}")


import argparse as _argp
_parser = _argp.ArgumentParser()
_parser.add_argument("--run", default="")
_parser.add_argument("--prefix", default="")
_args = _parser.parse_args()
_apply_overrides(_args.run, _args.prefix)

_c, _w, _v = build()
asyncio.run(main(_c, _w, _v))
