"""Read-only ceiling test for the two remaining proposals. No writes, no LLM calls.

Same discipline as consolidation_audit.py: before paying for an ingest, establish how
many questions the idea could POSSIBLY convert, and compare that to the measurement
noise (+/-6-7 questions per arm at one run each on this benchmark). An idea whose
ceiling sits under the noise floor can never be shown to work, however sound it is.

TEMPORAL — the proposal is to give a Fact `event_time` separate from `mentioned_at`,
plus planned/completed status. It can only help a question if the RIGHT date is already
in the retrieved text and the model chose a different one that is ALSO there; that is
the signature of "two dates present, no type to tell them apart". If the gold date is
absent from the retrieved text, no representation of time fixes it — the date was never
retrieved. If the gold date is present and the predicted one is not, the model invented
a date, which is generation, not representation.

COUNTING — the proposal is event-state plus citation dedup. It can only help if the
retrieved set already contains enough DISTINCT supporting memories to reach the gold
count. If it contains fewer, the miss is retrieval and no counting rule recovers it.

    python ceiling_audit.py
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
NOISE = 7          # +/- questions per arm, one run each

MONTHS = ("january february march april may june july august september october "
          "november december").split()
_M = "|".join(MONTHS)
DATE = re.compile(rf"\b(?:(?:{_M})\w*\s+\d{{1,2}}(?:,\s*\d{{4}})?|"
                  rf"\d{{1,2}}\s+(?:{_M})\w*(?:,?\s*\d{{4}})?|"
                  rf"(?:{_M})\w*\s+\d{{4}}|\b(?:19|20)\d{{2}})\b", re.I)
NUM = re.compile(r"\b(\d+)\b")
WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
           "once": 1, "twice": 2, "thrice": 3}
HOWMANY = re.compile(r"^how (many|much|long|often)", re.I)


def dates(text) -> set:
    """Normalised date-ish strings, so 'May 7, 2023' and '7 May 2023' compare equal."""
    out = set()
    for m in DATE.finditer(str(text or "")):
        s = m.group(0).lower().replace(",", "")
        toks = sorted(s.split())
        out.add(" ".join(toks))
    return out


def counts(text) -> set:
    t = str(text or "").lower()
    out = {int(x) for x in NUM.findall(t) if len(x) <= 3}
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", t):
            out.add(v)
    return out


def build() -> tuple:
    """Synchronous on purpose: MirixMemorySystem.__init__ calls asyncio.run internally,
    so constructing one inside a coroutine raises "cannot be called from a running
    event loop". Every client has to exist before the loop starts."""
    from mirix_memory_system import MirixMemorySystem

    judged = json.load(open(RESULTS))["llm_judge_results"]
    wrong = [r for r in judged if r.get("score") == 0]
    temporal = [r for r in wrong if str(r.get("category")) == "2"]
    counting = [r for r in wrong if HOWMANY.search(r["question"])]
    convs = sorted({r["sample_id"] for r in temporal + counting})
    clients = {c: MirixMemorySystem(
        user_id=PREFIX + c,
        mirix_config_path="/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml")
        for c in convs}
    print(f"temporal errors {len(temporal)}, counting errors {len(counting)}", flush=True)
    return clients, temporal, counting


async def run(clients, temporal, counting) -> None:
    sem = asyncio.Semaphore(6)

    async def fetch(r):
        async with sem:
            res = await clients[r["sample_id"]].client.search(
                user_id=PREFIX + r["sample_id"], query=r["question"],
                memory_type="all", search_method="embedding", limit=15)
        rows = [x for x in (res.get("results") or []) if isinstance(x, dict)]
        blob = " ".join(f"{x.get('summary') or ''} {x.get('details') or ''} "
                        f"{x.get('name') or ''}" for x in rows)
        return rows, blob

    # ---------------- temporal ----------------
    tstat = collections.Counter()
    tex = collections.defaultdict(list)

    async def one_t(r):
        rows, blob = await fetch(r)
        g, p, seen = dates(r["expected_answer"]), dates(r["predicted_answer"]), dates(blob)
        if not g:
            tstat["gold has no parseable date"] += 1
            return
        if not (g & seen):
            k = "gold date NOT in retrieved text -> retrieval, time typing cannot help"
        elif p & seen:
            k = "gold AND predicted date both present -> SELECTION, time typing could help"
        elif p:
            k = "gold present, predicted date absent -> model invented a date"
        else:
            k = "gold present, no date in answer at all"
        tstat[k] += 1
        tex[k].append(r)

    # ---------------- counting ----------------
    cstat = collections.Counter()
    cex = collections.defaultdict(list)

    async def one_c(r):
        rows, blob = await fetch(r)
        g, p = counts(r["expected_answer"]), counts(r["predicted_answer"])
        gv = min(g) if g else None
        if gv is None:
            cstat["gold has no parseable number"] += 1
            return
        # how many DISTINCT retrieved rows share content with the question?
        qt = {w for w in re.sub(r"[^a-z0-9 ]", " ", r["question"].lower()).split()
              if len(w) > 4}
        support = sum(1 for x in rows
                      if qt & {w for w in re.sub(r"[^a-z0-9 ]", " ",
                               f"{x.get('summary') or ''} {x.get('details') or ''}".lower()
                               ).split() if len(w) > 4})
        pv = min(p) if p else None
        direction = ("under" if pv is not None and pv < gv else
                     "over" if pv is not None and pv > gv else "no number")
        if support < gv:
            k = f"retrieved rows ({support}) < gold count ({gv}) -> retrieval"
        else:
            k = f"enough rows retrieved, model counted {direction} -> COUNTING could help"
        cstat[k] += 1
        cex[k].append((r, support, gv, pv))

    await asyncio.gather(*[one_t(r) for r in temporal],
                         *[one_c(r) for r in counting])

    def report(title, stat, total, examples=None):
        print(f"\n=== {title} — {total} errors ===")
        for k, n in stat.most_common():
            print(f"  {n:3d}  {k}")
        cap = sum(n for k, n in stat.items() if "could help" in k or "COULD" in k)
        verdict = ("worth building" if cap >= 2 * NOISE else
                   f"KILL: ceiling {cap} is under 2x the +/-{NOISE} noise floor")
        print(f"  -> ceiling {cap} questions   {verdict}")
        return cap

    tcap = report("TEMPORAL: event_time vs mentioned_at, planned/completed",
                  tstat, len(temporal))
    for r in tex["gold AND predicted date both present -> SELECTION, time typing could help"][:6]:
        print(f"     [{r['sample_id']}] {r['question'][:58]}")
        print(f"        gold {str(r['expected_answer'])[:28]!r}  answered "
              f"{str(r['predicted_answer'])[:56]!r}")
    ccap = report("COUNTING: event state + citation dedup", cstat, len(counting))
    for k in cex:
        if "COUNTING could help" not in k:
            continue
        for r, sup, gv, pv in cex[k][:6]:
            print(f"     [{r['sample_id']}] {r['question'][:54]}  gold={gv} answered={pv} "
                  f"rows={sup}")

    print(f"\n=== combined ceiling {tcap + ccap} questions on 1,540 "
          f"({(tcap + ccap) / 1540:.1%}) ===")
    print(f"    93% needs +61. Noise is +/-{NOISE} per arm at one run each.")


_clients, _temporal, _counting = build()
asyncio.run(run(_clients, _temporal, _counting))
