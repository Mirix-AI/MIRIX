"""Re-score our runs with HyperMem's judge, copied from their code rather than described.

HyperMem (ACL 2026) reports 92.73 on LoCoMo. Its grader is
EverMind-AI/HyperMem :: hypermem/main/stage6_eval.py :: locomo_grader, and its accuracy
prompt is BYTE-IDENTICAL to ours — same Hawaii/shell-necklace example, same contradictory
"explain then just return the label", same gpt-4o-mini, same temperature 0. Two things
differ, and both are ours:

    response_format   they do not set it; we set {"type": "json_object"}.
    system prompt     they send "You are an expert grader that determines if answers to
                      questions match a gold standard answer". We send none.

Neither difference matters, and the reason is worth recording because it is the opposite of
what it looks like. The prompt opens with "First, provide a short (one sentence) explanation
of your reasoning" and closes with "Just return the label CORRECT or WRONG in a json format".
gpt-4o-mini obeys the closing instruction. Run side by side on the same questions, both
configurations emit a bare {"label": "..."} and nothing else. NEITHER JUDGE REASONS — the
prompt countermands its own chain of thought, and response_format is not what suppresses it.
Measured agreement over 1540 questions is 98.8%, with 19 flips attributable to the system
prompt alone.

They also parse the label leniently: JSON first, then a regex for a JSON object embedded in
prose, then a bare CORRECT/WRONG in the text. With response_format removed, that fallback
chain is what makes free-form reasoning survivable.

Reproduced here exactly, so the comparison against 92.73 is a comparison of systems rather
than of graders. Absolute numbers under this judge are not comparable to the LoCoMo paper's
own metric, which is substring F1/BLEU-1 — no LLM judge is.

    python hypermem_judge.py --run clean_v724_qaonly
"""
import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

SYSTEM_PROMPT = """
    You are an expert grader that determines if answers to questions match a gold standard answer
    """

ACCURACY_PROMPT = """
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """


async def grade(client, question, gold_answer, response, retries=5):
    prompt = ACCURACY_PROMPT.format(
        question=question, gold_answer=gold_answer, response=response)
    for attempt in range(retries):
        try:
            api = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0,
            )
            content = api.choices[0].message.content
            try:
                label = json.loads(content)["label"]
            except json.JSONDecodeError:
                m = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]+"\s*[^{}]*\}', content)
                if m:
                    label = json.loads(m.group())["label"]
                elif "CORRECT" in content.upper() and "WRONG" not in content.upper():
                    label = "CORRECT"
                elif "WRONG" in content.upper() and "CORRECT" not in content.upper():
                    label = "WRONG"
                else:
                    raise ValueError(content[:200])
            return label.strip().lower() == "correct"
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                return None
            await asyncio.sleep((2 ** attempt) * 0.5)
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clean_v724_qaonly")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    path = f"/home/lj/code/MIRIX/evals/results/locomo/{a.run}/metrics.json"
    rows = [r for r in json.load(open(path))["llm_judge_results"]
            if str(r.get("category")) != "5"]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{a.run}: re-grading {len(rows)} with HyperMem's grader", flush=True)

    # Their protocol is majority-of-3, not a single call: stage6_eval.py:344 sets
    # num_runs = 3, :287 fires three graders per question concurrently, and :481 scores
    #     is_correct = true_count >= (num_runs / 2)
    # which for three runs means at least two must say CORRECT. temperature=0 is not
    # deterministic in practice, so this is variance reduction — and it SHARPENS, pushing
    # an item whose per-call probability of CORRECT is p to p**3 + 3*p**2*(1-p). Above 0.5
    # that is higher than p, below 0.5 lower. A single-call reproduction is therefore not
    # their protocol and cannot be compared to their 92.73.
    sem = asyncio.Semaphore(16)

    async def one(r):
        async with sem:
            votes = await asyncio.gather(*[
                grade(client, r["question"], r["expected_answer"], r.get("predicted_answer"))
                for _ in range(3)
            ])
            ok = [v for v in votes if v is not None]
            if not ok:
                return None
            return sum(1 for v in ok if v) >= (3 / 2)

    got = await asyncio.gather(*[one(r) for r in rows])
    ok = [(r, g) for r, g in zip(rows, got) if g is not None]
    new = sum(1 for _, g in ok if g)
    old = sum(1 for r, _ in ok if r.get("score") == 1)
    w2c = sum(1 for r, g in ok if r.get("score") == 0 and g)
    c2w = sum(1 for r, g in ok if r.get("score") == 1 and not g)
    print(f"\n  ours   {old}/{len(ok)} = {old/len(ok):.4f}")
    print(f"  theirs {new}/{len(ok)} = {new/len(ok):.4f}   ({new-old:+d})")
    print(f"  flipped WRONG->CORRECT {w2c}, CORRECT->WRONG {c2w}, "
          f"agreement {sum(1 for r,g in ok if (r.get('score')==1)==g)/len(ok):.1%}")
    print(f"  [HyperMem reports 92.73 on LoCoMo with this grader]")

    import collections
    CAT = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop"}
    d = collections.defaultdict(lambda: [0, 0])
    for r, g in ok:
        k = CAT.get(str(r.get("category")), "?")
        d[k][0] += 1
        d[k][1] += int(bool(g))
    print(f"\n  {'category':14s}{'n':>5s}{'ours':>9s}{'HyperMem':>10s}")
    HM = {"single-hop": .9608, "multi-hop": .9362, "temporal": .8972, "open-domain": .7083}
    for k in ("single-hop", "multi-hop", "temporal", "open-domain"):
        n, c = d[k]
        print(f"  {k:14s}{n:>5d}{c/n:>9.4f}{HM[k]:>10.4f}")


asyncio.run(main())
