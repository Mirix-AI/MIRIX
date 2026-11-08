"""Build a verbatim cold-fact index for LoCoMo, one entry per content-bearing turn.

The error triage on ab_graph_r1 put 60 of 169 wrong answers in "never written to the
store". Reading them, the losses are of two kinds and one of them is pure summarisation
damage: a nickname ("Jo"), a book title ("The Lean Startup"), an offhand joke, a painting's
description. The summariser judged them unimportant, so no memory row carries them and no
amount of retrieval work can recover them.

The raw turn always carries them. This writes the raw turns as a separate retrieval index
alongside the summarised store — the arrangement LongMemEval and Dense-X both argue for —
so the answerer can reach a verbatim line when the summary dropped its detail.

Differs from build_coldfacts.py, its LongMemEval sibling, in two ways that matter:

  * NO LLM. That one asks a model to extract literals from selected turns. Every LLM pass in
    this pipeline has been measured to lose or invent — 71% of quoted work-titles in the
    store appear nowhere in the source. A verbatim copy cannot hallucinate, and the detail
    we are chasing is exactly the kind an extractor discards.
  * Both speakers, not just the user. LoCoMo asks about either participant.

Each entry carries the speaker and the session date, because a bare quote is unusable for a
temporal question and the store's own rows are already dated. Photo captions are appended to
their turn rather than indexed separately — the caption is only meaningful with the remark
it accompanied.

ada-002 embeddings, matching _retrieve_coldfacts in task_agent.py and the store itself.
(mirix/embeddings.py pins ada-002 regardless of configuration; here that pin is what makes
the vectors comparable, so it is the right model rather than the bug.)

    python build_coldfacts_locomo.py --data data/locomo10.json --prefix clean-r1__
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/locomo10.json")
    ap.add_argument("--prefix", default="clean-r1__")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/MIRIX_eval"))
    ap.add_argument("--min-words", type=int, default=4)
    a = ap.parse_args()

    from openai import OpenAI
    from dotenv import load_dotenv
    import numpy as np
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    convs = json.load(open(a.data, encoding="utf-8"))
    total = 0
    for conv in convs:
        sid = conv["sample_id"]
        c = conv["conversation"]
        dates = {}
        for k, v in c.items():
            m = re.fullmatch(r"session_(\d+)_date_time", k)
            if m:
                dates[int(m.group(1))] = v

        facts = []
        for k, v in c.items():
            m = re.fullmatch(r"session_(\d+)", k)
            if not m or not isinstance(v, list):
                continue
            when = dates.get(int(m.group(1)), "")
            for t in v:
                text = " ".join(str(t.get("text") or "").split())
                cap = " ".join(str(t.get("blip_caption") or "").split())
                if len(text.split()) < a.min_words and not cap:
                    continue          # "Yeah!", "Thanks" — nothing to recover
                line = f"{t.get('speaker')}, on {when}, said: \"{text}\""
                if cap:
                    line += f" [shared a photo: {cap}]"
                facts.append({"fact": line, "ts": when, "dia_id": t.get("dia_id")})

        # embed in batches; ada-002 takes 2048 inputs per call but keep it modest
        vecs = []
        for i in range(0, len(facts), 256):
            batch = [f["fact"][:8000] for f in facts[i:i + 256]]
            r = client.embeddings.create(model="text-embedding-ada-002", input=batch)
            vecs.extend(d.embedding for d in r.data)
        arr = np.array(vecs, dtype="float32")
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        for f, e in zip(facts, arr):
            f["emb"] = [float(x) for x in e]

        path = os.path.join(a.out_dir, f"coldfacts_{a.prefix}{sid}.json")
        json.dump(facts, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        total += len(facts)
        print(f"{sid}: {len(facts)} facts -> {path}", flush=True)
    print(f"\n{total} verbatim turns indexed across {len(convs)} conversations")


if __name__ == "__main__":
    main()
