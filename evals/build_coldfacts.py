"""PoC: build a selective COLD-FACT index for a LongMemEval user from the RAW
conversation (not the summarized store). Recovers verbatim specifics — exact
numbers, quantities, prices, durations, specs, product/proper names — that
MIRIX's summarizing ingest drops. LongMemEval (ICLR'25) + Dense-X (EMNLP'24):
keep facts as a SEPARATE retrieval index alongside summaries, don't replace them.

Selective: only user turns that actually contain a literal are sent to the LLM,
so the fact count stays small (no Dense-X row explosion). Output: a JSON list of
{fact, ts, emb(ada-002)} the answerer retrieves from under MIRIX_COLDFACT.
"""
import ast
import json
import os
import re
import sys

from openai import OpenAI

USER = sys.argv[1] if len(sys.argv) > 1 else "longmem_s_0"
OUT = os.path.expanduser(f"~/MIRIX_eval/coldfacts_{USER}.json")

sys.path.insert(0, "/home/lj/code/MIRIX/evals")
from longmem_eval import load_longmem_s  # noqa: E402

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

item = load_longmem_s(limit=1)[0]
parsed = ast.literal_eval(item["context"])

# A turn is "literal-bearing" if it has a digit, price, %, or a WORD-number
# (three months, five sessions) — word-numbers were the biggest missed-fact gap.
LITERAL = re.compile(
    r"\b\d+\b|\$\d|\d+\s*%|"
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"first|second|third|fourth|fifth|couple|dozen|several)\b", re.I)


def chat_time(session):
    for m in session if isinstance(session, list) else []:
        if isinstance(m, dict):
            mt = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", str(m.get("content", "")))
            if mt:
                return mt.group(0)
    return None


# Collect literal-bearing USER turns with their session date.
turns = []
for session in parsed:
    msgs = session if isinstance(session, list) else session.get("messages", []) if isinstance(session, dict) else []
    ts = chat_time(session)
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if str(m.get("role", "")).lower() != "user":
            continue
        content = str(m.get("content", "")).strip()
        if content and LITERAL.search(content):
            turns.append((ts, content))

print(f"literal-bearing user turns: {len(turns)}", flush=True)

# Batch into ~16k-char groups, extract self-contained cold facts per batch.
PROMPT = (
    "From the USER's own messages below, extract every SPECIFIC personal fact the user "
    "states about themselves — prioritizing exact NUMBERS, quantities, prices, DURATIONS, "
    "dates, speeds, measurements, POSSESSION COUNTS, and product/proper NAMES. Preserve "
    "word-numbers too (e.g. 'collecting cameras for three months', 'attended five sessions', "
    "'owns four bikes: road, mountain, commuter, hybrid'). Write each as ONE short "
    "self-contained sentence that PRESERVES THE EXACT VALUE verbatim (e.g. 'The user "
    "brought 7 shirts to Costa Rica', 'The user upgraded to 500 Mbps internet', 'The user "
    "has been collecting vintage cameras for three months', 'The user made a lemon "
    "poppyseed cake for a colleague'). Only facts about the user's own life, "
    "possessions, activities, and preferences. Skip generic/assistant content. Output one "
    "fact per line, no numbering.\n\nUSER MESSAGES:\n")

facts = []
batch, blen = [], 0
def flush(batch):
    if not batch:
        return
    txt = "\n".join(f"[{ts or '?'}] {c}" for ts, c in batch)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "user", "content": PROMPT + txt}])
    for line in resp.choices[0].message.content.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if len(line) > 8:
            facts.append(line)

for ts, c in turns:
    if blen + len(c) > 16000:
        flush(batch); batch, blen = [], 0
    batch.append((ts, c)); blen += len(c)
flush(batch)

# Dedup + embed (ada-002, matching the store's space).
seen, uniq = set(), []
for f in facts:
    k = f.lower()
    if k not in seen:
        seen.add(k); uniq.append(f)
print(f"extracted {len(facts)} facts -> {len(uniq)} unique; embedding...", flush=True)

embs = []
for i in range(0, len(uniq), 256):
    chunk = uniq[i:i + 256]
    r = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
    embs.extend(d.embedding for d in r.data)

index = [{"fact": f, "emb": e} for f, e in zip(uniq, embs)]
json.dump(index, open(OUT, "w"))
print(f"wrote {OUT}  ({len(index)} cold facts)")
# sanity: are the known targets captured?
for probe in ["500 mbps", "7 shirts", "poppyseed", "50 hours", "three months", "420"]:
    hit = [f for f in uniq if probe in f.lower()]
    print(f"  probe '{probe}': {hit[:1]}")
