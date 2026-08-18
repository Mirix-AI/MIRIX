"""Direct retrieval-quality metric for LoCoMo conv-26 — no answerer involved.

Why: every retrieval change so far has been judged through end-to-end QA, where the
answerer's own variance is +/-1.5 questions and a 25-minute ingest sits in front of
each measurement. This asks the narrower question that retrieval work actually needs
answered — "did the evidence come back at all?" — in a couple of minutes, with no LLM
in the loop and therefore no answerer noise.

TWO LEVELS, because one of them lies. Session-level recall asks only whether some
memory from the gold SESSION came back; on the live store it reads a uniform 0.940
across conversations while QA ranges 0.844-0.942, and cross-referencing showed 23 of
60 wrong answers had the gold session present but NOT the answer-bearing memory. The
session metric cannot see those. Answer-support recall asks the narrower question —
does the returned TEXT actually contain the gold answer's content — and needs no
turn-level alignment, only the gold answer that already exists.

Report both. The GAP between them is exactly the "found the region, missed the
evidence" mass, and it is the quantity a retrieval change has to move.

Answer-support is a proxy with a known bias: it under-reports on gold answers that are
abstractive ("Somewhat, but not extremely religious") or a single token ("2022"), where
containment is a poor test of support. Treat it as a lower bound, and read the two
numbers together rather than either alone.

How the gold is derived: LoCoMo tags each question with dialogue ids like 'D1:3'
(session 1, turn 3). main_eval ingests one chunk per session in order, and the server
stamps each memory's source_refs with a per-user monotonic chunk counter, so
session k is chunk_id k-1. A question is RECALLED when at least one returned memory
carries a gold chunk_id. That is session-level, not turn-level: coarser than ideal,
but exact, and it is the finest granularity the store actually records.

    python retrieval_recall.py --user conv-26 --limit 199
    python retrieval_recall.py --user conv-26 --graph-only     # graph path alone

Requires the server to be running against the store you want to measure.
"""
import argparse
import ast
import asyncio
import collections
import json
import os
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

DATA = "/home/lj/code/MIRIX/evals/data/locomo10.json"
DIA = re.compile(r"D(\d+):(\d+)")
_STOP = set("the a an and or of to in on at for with from that this it is was were be "
            "been have has had do does did what when where who how why which their her "
            "his its she he they them about into over under after before not".split())


def _content(text) -> set:
    """Content tokens: >3 chars, not a stopword. Also keeps bare years and counts,
    which are frequently the whole gold answer."""
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in _STOP}


def gold_chunks(evidence) -> set:
    """'[\"D1:3\", \"D2:8\"]' -> {0, 1}  (session k -> chunk_id k-1)."""
    if isinstance(evidence, str):
        try:
            evidence = ast.literal_eval(evidence)
        except Exception:  # noqa: BLE001
            evidence = [evidence]
    out = set()
    for item in evidence or []:
        m = DIA.search(str(item))
        if m:
            out.add(int(m.group(1)) - 1)
    return out


def build(args):
    from mirix_memory_system import MirixMemorySystem
    return MirixMemorySystem(
        user_id=args.user,
        mirix_config_path="/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="conv-26",
                    help="PG/Neo4j user_id, e.g. v723full-r1__conv-26")
    ap.add_argument("--sample", default="",
                    help="LoCoMo sample_id the questions come from; defaults to the "
                         "trailing conv-NN of --user")
    ap.add_argument("--limit", type=int, default=199)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--graph-only", action="store_true",
                    help="measure the graph retriever alone instead of /search")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="", help="write per-question detail to this JSON")
    return ap.parse_args()


async def main(a, ms) -> None:
    with open(DATA) as fh:
        items = json.load(fh)
    sample = a.sample or a.user.split("__")[-1]
    item = next((x for x in items if x.get("sample_id") == sample), None)
    if item is None:
        raise SystemExit(f"no LoCoMo sample {sample!r}; have "
                         f"{[x['sample_id'] for x in items]}")
    for i, q in enumerate(item["qa"], start=1):
        q["_qi"] = i          # main_eval numbers questions 1..N over the FULL qa list
    qa = [q for q in item["qa"] if q.get("evidence") and q.get("answer") is not None][: a.limit]

    # memory id -> set(chunk_id), read once straight from PG
    from sqlalchemy import text as sa_text
    from mirix.server.server import db_context
    id2chunk: dict = {}
    async with db_context() as session:
        for table in ("episodic_memory", "semantic_memory"):
            rows = (await session.execute(sa_text(
                f"SELECT id, source_refs FROM {table} "
                f"WHERE user_id = :u AND NOT is_deleted"), {"u": a.user})).fetchall()
            for mid, refs in rows:
                cs = set()
                for r in (refs or []):
                    if isinstance(r, dict) and r.get("chunk_id") is not None:
                        cs.add(int(r["chunk_id"]))
                id2chunk[mid] = cs
    print(f"{len(id2chunk)} memories, "
          f"{len({c for cs in id2chunk.values() for c in cs})} distinct chunks", flush=True)

    hits = collections.Counter()          # session-level
    sup = collections.Counter()           # answer-support level
    tot = collections.Counter()
    misses = []
    per_q: dict = {}       # question_index -> {recalled, gold, got, category}
    sem = asyncio.Semaphore(6)

    async def one(q):
        gold = gold_chunks(q.get("evidence"))
        if not gold:
            return
        cat = str(q.get("category"))
        async with sem:
            if a.graph_only:
                from mirix.services.graph_retriever_dispatcher import GraphRetrieverDispatcher
                rows = await GraphRetrieverDispatcher().retrieve_rows(
                    query=q["question"], user_id=a.user,
                    agent_state=ms.client._agent_state if hasattr(ms.client, "_agent_state") else None,
                    max_items_per_kind=a.k)
                got_ids = [r.id for r in rows]
            else:
                res = await ms.client.search(user_id=a.user, query=q["question"],
                                             memory_type="all", search_method="embedding",
                                             limit=a.k)
                got_ids = [r.get("id") for r in (res.get("results") or []) if r.get("id")]
        got_chunks = set()
        for mid in got_ids:
            got_chunks |= id2chunk.get(mid, set())
        ok = bool(gold & got_chunks)

        # answer-support: does the returned TEXT carry the gold answer's content?
        blob = " ".join(
            f"{x.get('summary') or ''} {x.get('details') or ''} {x.get('name') or ''}"
            for x in (res.get("results") or []) if isinstance(x, dict)
        ) if not a.graph_only else " ".join(
            f"{getattr(r, 'summary', '')} {getattr(r, 'details', '')}" for r in rows)
        gtok = _content(q.get("answer"))
        supported = bool(gtok) and len(gtok & _content(blob)) / len(gtok) >= 0.6
        tot[cat] += 1
        tot["ALL"] += 1
        if ok:
            hits[cat] += 1
            hits["ALL"] += 1
        if supported:
            sup[cat] += 1
            sup["ALL"] += 1
        else:
            misses.append((q.get("_qi"), cat, q["question"][:70],
                           sorted(gold), sorted(got_chunks)[:6]))
        per_q[str(q.get("_qi"))] = {"recalled": ok, "category": cat,
                                    "supported": supported,
                                    "gold": sorted(gold), "got": sorted(got_chunks),
                                    "question": q["question"]}

    await asyncio.gather(*[one(q) for q in qa])

    tag = a.label or ("graph-only" if a.graph_only else "search (graph+flat)")
    print(f"\n=== retrieval recall@{a.k} — {tag} — user={a.user} ===")
    print(f"  {'cat':<7s}{'session':>16s}{'answer-support':>18s}{'gap':>9s}")
    for cat in sorted(tot, key=lambda c: (c == "ALL", c)):
        n, h, sp = tot[cat], hits[cat], sup[cat]
        if cat == "ALL":
            print("-" * 52)
        print(f"  {cat:<7s}{h:8d}/{n:<4d} {h/n:.3f}{sp:8d}/{n:<4d} {sp/n:.3f}"
              f"{(h - sp) / n:+9.3f}")
    print(f"\n{len(misses)} misses (gold session never returned):")
    for qi, cat, q, gold, got in misses[:15]:
        print(f"  q{qi} [cat {cat}] {q}\n        gold chunks {gold}, got {got}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"user": a.user, "k": a.k, "label": tag,
                       "per_question": per_q,
                       "totals": {c: {"session": hits[c], "support": sup[c],
                                      "n": tot[c]} for c in tot}}, fh, indent=1)
        print(f"\nper-question detail -> {a.out}")


_args = parse_args()
asyncio.run(main(_args, build(_args)))
