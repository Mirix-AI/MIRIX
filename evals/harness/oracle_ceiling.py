"""If retrieval were perfect, how much would it be worth?

Four measurements this week say the answerer gets worse when given more evidence, and the
last one — v7.26, widening candidate admission, mean -18 over four paired comparisons —
rescued 31-38 questions while breaking 51-55. The retrieval bucket is real (38 of 169 errors
have the answer in the store and never retrieved), but admitting more candidates is not the
way in: the correct row has to WIN, not merely be present.

Before anyone builds a better ranker, this measures what a perfect one would be worth. Each
question is answered from the fifteen store rows that best match its own GOLD answer — an
oracle no real system can have, so whatever it scores is an upper bound on ranking work.

Scored on both sides, because an oracle context is also a REPLACED context: the control
questions are currently answered correctly from ordinary retrieval, and if the oracle's
fifteen rows break them, that cost belongs in the total.
"""
import collections, json, os, random, re, subprocess, sys
sys.path.insert(0, "/home/lj/code/MIRIX/evals")
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from llm_judge import evaluate_llm_judge, assert_canonical_prompt

assert_canonical_prompt()
cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them what when which who how "
           "why".split())


def C(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return {x for x in w if (len(x) > 2 or x.isdigit()) and x not in STOP}


q = ("SELECT user_id, coalesce(summary,'')||' — '||coalesce(details,'') FROM episodic_memory "
     "WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' UNION ALL SELECT user_id, "
     "coalesce(name,'')||' — '||coalesce(details,'') FROM semantic_memory "
     "WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%';")
out = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                      "-d", "mirix_locomo_clean_r1", "-At", "-F", "\t", "-c", q],
                     capture_output=True, text=True,
                     env={**os.environ, "PGPASSWORD": "mirix"},
                     cwd="/home/lj/MIRIX_eval").stdout
rows = collections.defaultdict(list)
for line in out.splitlines():
    p = line.split("\t")
    if len(p) >= 2 and p[1].strip():
        rows[p[0]].append(" ".join(p[1].split()))

D = "/home/lj/code/MIRIX/evals/results/locomo/ab_graph_r1"
J = [x for x in json.load(open(D + "/metrics.json"))["llm_judge_results"]
     if str(x.get("category")) != "5"]
errs = [x for x in J if x.get("score") == 0]
random.seed(11)
ctrl = random.sample([x for x in J if x.get("score") == 1], 150)

SYS = ("Answer the question using ONLY the memories below. "
       "Give the minimal direct answer and nothing else.")


def oracle_ctx(x, k=15):
    g, qs = C(x["expected_answer"]), C(x["question"])
    sc = []
    for r in rows.get("clean-r1__" + x["sample_id"], []):
        c = C(r)
        if c:
            sc.append((len(g & c) / max(len(g), 1) * 2 + len(qs & c) / max(len(qs), 1), r))
    sc.sort(reverse=True)
    return "\n".join(f"- {r}" for _, r in sc[:k])


def one(x):
    p = f"{SYS}\n\nMemories:\n{oracle_ctx(x)}\n\nQuestion: {x['question']}"
    try:
        r = cl.chat.completions.create(model="gpt-4.1-mini",
                                       messages=[{"role": "user", "content": p}],
                                       temperature=0, seed=42, max_completion_tokens=128)
        a = (r.choices[0].message.content or "").strip()
        return int(bool(evaluate_llm_judge(x["question"], x["expected_answer"], a))) if a else 0
    except Exception:  # noqa: BLE001
        return 0


with ThreadPoolExecutor(max_workers=12) as ex:
    e = sum(ex.map(one, errs))
    c = sum(ex.map(one, ctrl))
loss = 1 - c / len(ctrl)
print(f"oracle: the 15 store rows that best match each question's OWN gold answer\n")
print(f"  currently wrong  {len(errs)}  ->  {e} rescued ({e/len(errs):.0%})")
print(f"  currently right  {len(ctrl)}  ->  {c} still right ({c/len(ctrl):.0%})")
print(f"\n  extrapolated net over 1540: {e - round(loss*1371):+d} questions")
print(f"  (control loss {loss:.1%} applied to the 1371 currently right)")
