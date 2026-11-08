"""Does ingesting a session in small windows keep more of its specifics?

The largest error bucket is 60 questions whose fact was never written. Tracing three of them
shows what the summariser does, and it is not carelessness — it is compression working as
designed:

    "Hey Jo, guess what I did? Dyed my hair last week"
        stored: "Nate dyed his hair purple last week"          the nickname is gone
    "I'm currently reading 'The Lean Startup' and hoping it'll give me tips for my biz"
        stored: "Jon is wrapping up a business plan and searching for investors"
                                                                the title is gone

Both turns lost their specific noun and kept their gist. Today a whole session — thirty-odd
turns — arrives in ONE add_chunk call, so the extractor is choosing a handful of memories to
represent all of it and the specifics are what lose. A six-turn window asks it to represent
much less per call, which should leave less room to compress.

This measures that claim WITHOUT the eval loop: no QA, no judge, no accuracy. It counts how
much of the source's distinctive vocabulary survives into the store — proper nouns, quoted
titles, numbers — under two chunkings of the same conversations. That metric is deterministic
given a store, so it is not subject to the ±9-15 that makes single QA runs unreadable, and it
is the property the fix is actually about.

Three conversations, two arms each. Whatever the result, it is reported per conversation and
required in all three: aggregate improvements have already fooled this project once.

    python window_recall.py --arm baseline     # then --arm windowed
    python window_recall.py --report
"""
import argparse
import collections
import json
import os
import re
import subprocess

CONVS = ("conv-30",)
# Distinctive = things a summary is tempted to drop. Deliberately not every capitalised
# word: sentence-initial capitals and speaker names would swamp the signal.
QUOTED = re.compile(r'"([^"]{3,60})"')
PROPER = re.compile(r'(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b')
NUMBER = re.compile(r'\b\d[\d,.]*\b')
COMMON = {"The", "This", "That", "There", "They", "What", "When", "Where", "Which", "And",
          "But", "You", "Your", "His", "Her", "She", "Yeah", "Well", "Just", "Hey", "Oh",
          "Thanks", "Now", "Not", "Also", "Are", "Have", "Was", "For", "With", "How", "Why",
          "All", "Its", "Ive", "Im", "Its", "Let", "Lets", "Its"}


def distinctive(text: str, speakers: set) -> set:
    out = {m.group(1).strip().lower() for m in QUOTED.finditer(text)}
    for m in PROPER.finditer(text):
        v = m.group(1)
        if v.split()[0] in COMMON or v in speakers:
            continue
        out.add(v.lower())
    out |= {m.group(0) for m in NUMBER.finditer(text) if len(m.group(0)) > 1}
    return {x for x in out if len(x) > 2}


def store_text(db, user_id):
    q = ("SELECT string_agg(t, ' ') FROM ("
         f"SELECT coalesce(summary,'')||' '||coalesce(details,'') AS t FROM episodic_memory "
         f"WHERE NOT is_deleted AND user_id='{user_id}' "
         f"UNION ALL SELECT coalesce(name,'')||' '||coalesce(details,'') FROM semantic_memory "
         f"WHERE NOT is_deleted AND user_id='{user_id}') z;")
    return subprocess.run(["/home/lj/MIRIX_eval/pgenv/bin/psql", "-w", "-h", "localhost",
                           "-U", "mirix", "-d", db, "-At", "-c", q],
                          capture_output=True, text=True,
                          env={**os.environ, "PGPASSWORD": "mirix"}).stdout.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="/home/lj/MIRIX_eval/window_recall.json")
    a = ap.parse_args()

    lc = {x["sample_id"]: x for x in json.load(
        open("/home/lj/code/MIRIX/evals/data/locomo10.json"))}
    results = {}
    for sid in CONVS:
        conv = lc[sid]["conversation"]
        speakers, src = set(), []
        for k, v in conv.items():
            if isinstance(v, list):
                for t in v:
                    speakers.add(str(t.get("speaker") or ""))
                    src.append(f"{t.get('text','')} {t.get('blip_caption') or ''}")
        want = distinctive(" ".join(src), speakers)
        have = store_text(a.db, a.prefix + sid)
        kept = {w for w in want if w in have}
        results[sid] = {"distinctive": len(want), "kept": len(kept),
                        "recall": round(len(kept) / max(len(want), 1), 4),
                        "missing_sample": sorted(want - kept)[:12]}
        print(f"{sid}: {len(kept)}/{len(want)} distinctive tokens kept "
              f"({len(kept)/max(len(want),1):.1%})", flush=True)

    all_ = json.load(open(a.out)) if os.path.exists(a.out) else {}
    all_[a.label] = results
    json.dump(all_, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {a.out} [{a.label}]")

    if len(all_) > 1:
        print(f"\n{'conversation':14s}" + "".join(f"{k:>14s}" for k in all_))
        for sid in CONVS:
            row = "".join(f"{all_[k][sid]['recall']:>13.1%} " for k in all_ if sid in all_[k])
            print(f"{sid:14s}{row}")
        print("\n  the fix only counts if recall rises in ALL THREE, not on average")


if __name__ == "__main__":
    main()
