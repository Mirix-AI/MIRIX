"""Build a compact, structured PERSONA profile for a LongMemEval user from the
existing memory store (no re-ingest). Targets the preference-synthesis questions
(Q7/Q11/Q44...) whose gold answers require grounding advice in the user's own
history — hobbies, past successes, stated likes/dislikes, work/social situation.

The persona is a pre-synthesized "who is this user" summary so the answerer does
not have to reconstruct it from scattered search hits on open-ended "any tips?"
questions. Written to persona_<user>.txt; TaskAgent injects it when MIRIX_PERSONA
is set. LaMP-PAG / PersonaAgent style.
"""
import os
import sys
import psycopg2
from openai import OpenAI

USER = sys.argv[1] if len(sys.argv) > 1 else "longmem_s_0"
PG_DB = os.environ.get("MIRIX_PG_DB") or os.environ.get("PG_DB", "mirix_lm114_pm")
OUT = os.path.expanduser(f"~/MIRIX_eval/persona_{USER}.txt")

conn = psycopg2.connect(host="localhost", port=5432, user="mirix", password="mirix", dbname=PG_DB)
cur = conn.cursor()
# User's own statements (episodic actor='user') + stable facts (semantic) = the
# raw material for a persona. Assistant turns are advice, not the user's identity.
cur.execute("SELECT occurred_at, summary FROM episodic_memory WHERE user_id=%s AND is_deleted=false "
            "AND actor='user' ORDER BY occurred_at", (USER,))
ep = cur.fetchall()
cur.execute("SELECT name, summary FROM semantic_memory WHERE user_id=%s AND is_deleted=false", (USER,))
sem = cur.fetchall()
cur.close(); conn.close()

lines = [f"[{str(o)[:10]}] {s}" for o, s in ep if s] + [f"{n}: {s}" for n, s in sem if n]
context = "\n".join(lines)

PROMPT = (
    "Below are a user's own statements over time plus stable facts about them. Build a "
    "concise PERSONA PROFILE that captures WHO THIS USER IS, so an assistant can give "
    "personalized advice grounded in their history (not generic tips).\n\n"
    "Organize by life domain (e.g. Food & Baking, Art & Creativity, Fitness, Work & "
    "Career, Social & Relationships, Travel, Hobbies, Home & Pets, Reading, Finance). "
    "For each domain that applies, capture in 1-3 tight bullets:\n"
    "  - what they DO / are into (recurring activities, tools, brands they use)\n"
    "  - notable PAST experiences / successes / projects (e.g. 'baked a lemon poppyseed "
    "cake that went well', 'did a 30-day painting challenge')\n"
    "  - stated PREFERENCES: likes, dislikes, constraints, what they value\n"
    "Be specific and name concrete things (recipes, apps, accounts, places, people). "
    "Skip domains with no signal. Keep the whole profile under ~450 words. Output plain "
    "text with domain headers and bullets, no preamble.\n\n"
    "USER STATEMENTS & FACTS:\n" + context
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
resp = client.chat.completions.create(
    model="gpt-4.1-mini", temperature=0,
    messages=[{"role": "user", "content": PROMPT}],
)
persona = resp.choices[0].message.content.strip()
with open(OUT, "w", encoding="utf-8") as f:
    f.write(persona)
print(f"wrote {OUT}  ({len(persona)} chars, from {len(ep)} user-episodic + {len(sem)} semantic)")
print("=" * 60)
print(persona)
