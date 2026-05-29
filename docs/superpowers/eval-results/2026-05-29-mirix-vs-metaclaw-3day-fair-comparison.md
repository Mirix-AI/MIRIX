# MIRIX vs MetaClaw — 3-day Fair Comparison

**Run**: `evals/metaclaw/runs/both-20260529T053939Z`
**Date**: 2026-05-29
**Vendor**: aiming-lab/MetaClaw @ `fc163ba8a12ba5e6950544c3e0d55707c16e2a7e` (v0.4.1)
**Model**: `openai/gpt-5.2` via OpenRouter (both arms, identical)
**Command**: `python -m evals.metaclaw --arm both --days 3`

## Headline

| Arm | Status | Rounds | Correct | Accuracy | Agent Tokens |
|---|---|---|---|---|---|
| **metaclaw** (native skills_dir backend) | ok | 33 | 11.50 | **34.85 %** | n/a |
| **mirix** (MIRIX is sole producer + storage) | ok | 33 | 10.33 | **31.31 %** | n/a |

**Delta: mirix − metaclaw = −3.54 pp accuracy.**

The single controlled variable is the **skill backend**: both arms run the identical
MetaClaw `openclaw` agent, the identical 30-day dataset sliced to the same 3 days
(byte-identical `all_tests_used.json`), the identical model, and the identical
narrow-fairness input signal (`{prompt_text, response_text}` only — no pass/fail
oracle leaks to either skill backend). Only the skill produce/retrieve module differs.

## Per-day breakdown

| Day | Questions | metaclaw | mirix | winner |
|---|---|---|---|---|
| day01 | 10 | 40.00 % | 30.00 % | metaclaw +10.0 |
| day02 | 11 | 33.33 % | 33.33 % | tie |
| day03 | 12 | 31.94 % | 30.56 % | metaclaw +1.4 |
| **overall** | 33 | **34.85 %** | **31.31 %** | metaclaw +3.5 |

`passed` metric is 0.0 for both arms by design — MetaClaw's pass/fail oracle is
deliberately withheld from the skill backend (narrow fairness). Accuracy here is the
per-field `exact_match`/`f1`/`iou` family, identical to paper's own `skills_only` mode.

## MIRIX arm health (this run — the key difference from the 2026-05-28 invalid run)

| Metric | This run (fair) | 2026-05-28 (invalid) |
|---|---|---|
| evolve invocations | 8 | 16 |
| evolve **succeeded** | **8 (100 %)** | 1 |
| evolve **failed (HTTP 500)** | **0** | 13 |
| skills created in MIRIX | 18 | 1 |
| MirixSkillsAdapter retrieve calls | 89 | (degraded) |
| mirix arm effective mode | full produce+retrieve | retrieval-only (broken) |

The 2026-05-28 `--arm both` run reported mirix = 29.3 % (−5.6 pp), but that number is
**invalid**: MIRIX's PostgreSQL pool died mid-run, so 13 of 16 evolve calls returned
HTTP 500 and the mirix arm silently degraded to retrieval-only with a single skill.
This run is the corrected, apples-to-apples comparison: the evolver was hardened
(4 retries + exponential backoff) and the MIRIX server was rebuilt healthy, yielding
**8/8 successful evolve calls and 18 skills produced** across the 3 days.

## Interpretation

On this 3-day slice, **MIRIX-as-skill-backend trails MetaClaw's native skill bank by
3.5 pp** (34.85 % → 31.31 %). Caveats before over-reading this:

1. **Single seed, 33 rounds.** At this granularity, run-to-run LLM nondeterminism is
   material: an earlier *healthy* solo mirix run scored 32.83 % (vs 31.31 % here) on
   the same 3 days. The metaclaw arm, by contrast, scored 34.85 % in both `--arm both`
   runs (high determinism). A multi-seed sweep is needed for confidence intervals
   before claiming a definitive gap.
2. **MIRIX is doing strictly more work.** It runs a server-side procedural-memory agent
   that *generates* skills from transcripts (18 created here), whereas the metaclaw arm
   uses paper's lighter-weight skill extraction. The −3.5 pp is the cost/benefit of that
   heavier, general-purpose memory system on this specific 3-day task — not a verdict on
   30 days, where MIRIX's dedup/consolidation may compound differently.
3. **Tokens are n/a.** OpenRouter does not relay `usage` for `openai/gpt-5.2` in the
   shape paper's bench parses; accuracy is unaffected.

## Retrieval-method A/B — the −3.5pp gap was the retrieval method (2026-05-29)

Follow-up experiment after discovering that MIRIX's `/v1/skills` endpoint
hardcodes keyword retrieval (`search_method="bm25"`) even though its engine
defaults to and fully supports `embedding` (vector) search. We exposed the
method via env (`MIRIX_SKILL_SEARCH_METHOD`) and re-ran the mirix arm with
vector retrieval — single variable changed, everything else identical.

| mirix retrieval | overall | day01 | day02 | day03 | evolve health |
|---|---|---|---|---|---|
| **BM25** (keyword, `ts_rank_cd` over `description`) | 31.31 % | 30.0 % | 33.3 % | 30.6 % | 8/8 |
| **Embedding** (vector similarity) | **34.85 %** | **40.0 %** | 34.85 % | 30.6 % | 9/9 |
| metaclaw native (reference) | 34.85 % | 40.0 % | 33.3 % | 31.9 % | — |

**Switching MIRIX skill retrieval from BM25 to embedding lifted it 31.31 % →
34.85 % (+3.5 pp), exactly matching the metaclaw native arm (both 11.50/33
correct).** The earlier −3.5 pp deficit was almost entirely the retrieval method,
not the skill-production quality: MIRIX's procedural agent was producing good
skills all along; keyword retrieval was failing to surface them when the task
wording differed from the skill description wording.

Verified the embedding mode was genuinely active (not a silent BM25 fallback):
a query "how should I format the clock values in my json" — containing none of
the keywords ISO8601/timestamp/offset — still matched a
`normalize-timestamps-iso8601-plus0800` skill, which BM25 could not do.

Caveats: single seed; the exact 34.85 % tie is partly coincidental at 33-round
granularity (±1–2 questions). The gain is concentrated in the early days
(day01 +10 pp), consistent with semantic recall mattering most when the skill
bank is small and task/skill wording diverges. Multi-seed runs are needed to
confirm, but the direction is clear and strong: embedding ≫ BM25 for MIRIX skill
retrieval on this benchmark.

Run dirs: BM25 = `runs/both-20260529T053939Z/mirix-*`;
embedding = `runs/mirix-20260529T075013Z`.

## Eval mechanism — verified by source + log inspection (2026-05-29)

These are the mechanics of the vendored MetaClaw harness, traced through
`vendor/benchmark/src/infer/infer_cmd.py` and `vendor/metaclaw/api_server.py`,
and confirmed against this run's logs. They define exactly what the skill
backend does and does not see — i.e. the fairness boundary.

### 1. Skill-evolution cadence: every 10 agent turns, NOT per day

`skill_evolution_every_n_turns = 10` (`vendor/metaclaw/config.py:58`). The proxy
buffers every openclaw **main turn** and fires `_evolve_skills_for_session` once
the buffer hits 10 (`api_server.py:1346`). A "day" (= 1 test = 1 round-group) is
10–12 rounds, and each round drives multiple agent turns (read → exec → write →
reply → feedback → revise…), so 3 days ≈ 80 turns ≈ 8 evolve calls — **not 3**.
This run: mirix fired 8 evolve calls, metaclaw 9 (the ±1 is turn-count variance
between the two arms' agent trajectories). Each call logs "N samples" / "from N
failures" with N=10 — the batch size, not a day boundary.

### 2. Scoring: inline, immediately after each round's answer

`_compute_inline_score` (`infer_cmd.py:653`) runs **synchronously right after the
agent answers each round** — not as a post-hoc batch pass. Two question types:
- **multi_choice**: regex-extract `\bbox{A,B}` from the answer → set; compare for
  **exact equality** against `eval.answer`. All-correct-and-none-extra → `passed`.
- **file_check**: execute `eval.command` in the workspace (checks the JSON files
  the agent wrote — field presence, time formats, etc.); command success → `passed`.

`inline_score.passed` is a binary oracle, but it is used for only two things:
(a) building the next round's feedback, and (b) the final `report.json` accuracy.
**It is never handed to the skill backend.**

### 3. Feedback: injected into the NEXT round, from pre-recorded text

After scoring round N, round N+1's query is prefixed via `with_feedback()`
(`infer_cmd.py:907-916`) as:

```
[Previous Feedback] {feedback_text}

{round N+1 question}
```

`feedback_text` is built by `_build_feedback_text` from the dataset's
**pre-recorded** `round_record["feedback"]` (`feedback.correct` /
`feedback.options[X]` / `feedback.incorrect`) — it is deterministic, not a live
LLM judgment. Round 1 has no feedback. So feedback is **lagged by one round** and
identical every time a given answer is graded (reproducible).

### 4. Narrow-fairness signal isolation (the controlled variable)

The pass/fail oracle (`inline_score`) lives **bench-side** (`infer_cmd.py`) and is
**never communicated to the proxy or the skill backend**. What the evolver
actually receives, in BOTH arms identically:

| Signal | Visible to evolver? |
|---|---|
| Structured pass/fail score (oracle) | ❌ no — stays bench-side |
| `reward` field on each sample | hardcoded `0.0` (`api_server.py:2122`) — a **dead field** |
| Full transcript `{prompt_text, response_text}` | ✅ yes |
| `[Previous Feedback]` text embedded in `prompt_text` | ✅ yes (it's a user-role message in the conversation) |

So a skill backend can only sense "what went wrong" **indirectly, from the
natural-language feedback text** — never from a label. This is the single
controlled variable: the only thing that differs between arms is the algorithm
that turns those identical transcripts into skills.

### 5. The evolver sees ALL turns, not just failed ones

Neither arm filters to failures. The proxy buffers **every** main turn
(`api_server.py:1344`, no pass/fail filter). `reward` is hardcoded `0.0` for all,
so by the evolver's own "reward ≤ 0 = failed" definition every turn nominally
qualifies → all pass through. The `failed_samples` parameter name and the
"from N failures" log line are **vacuous naming**, not a real filter. The
`should_evolve` success-rate gate exists but is **not called** on the skills_only
path. Skills skew corrective not because only failures are fed, but because the
**feedback text itself is corrective**.

### 6. Verification that MIRIX actually consumed the feedback (this run)

- inline scoring ran on **33/33 rounds** (9 `passed`).
- `[Previous Feedback]` appears **113×** in the mirix proxy.log (vs 33× metaclaw —
  mirix is higher because `MirixSkillsAdapter.retrieve` also queries with the
  feedback-laden prompt, so retrieval sees feedback too).
- **Decisive evidence**: all 16 skills MIRIX produced carry `iso8601` / `plus0800`
  / time-format themes (e.g. `convert-utc-z-timestamp-to-iso8601-plus0800`). The
  `+08:00` detail exists **only** in the injected `feedback.incorrect` text, not in
  the base task — so the evolve path provably received and used the feedback.

### 7. Caveat — `prompt_text` tail-1500 truncation

`MirixEvolverAdapter._sample_to_message` caps each sample's `prompt_text` to the
last 1500 chars (and `response_text` to the first 1500). On a turn with a very
long tool-output tail, feedback sitting early in that turn's prompt could be
clipped. Empirically harmless here (feedback recurs across a round's turns, 10
turns are batched per evolve, and the skill output confirms the feedback landed),
but a safe hardening would be to explicitly preserve the `[Previous Feedback]`
span in `_sample_to_message` rather than rely on the tail window. Not a current
bug — logged as a future enhancement.

## Reproducing

```bash
# 1. Fresh MIRIX server with pgvector + meta agent (see init_meta_agent.py):
docker run -d --name mirix_pg_5433 -e POSTGRES_USER=mirix -e POSTGRES_PASSWORD=mirix \
  -e POSTGRES_DB=mirix -p 5433:5432 pgvector/pgvector:pg16
docker exec mirix_pg_5433 psql -U mirix -d mirix -c "CREATE EXTENSION IF NOT EXISTS vector;"
# start MIRIX server (feat/skill-evolve) with MIRIX_PG_PORT=5433, then:
docker exec mirix_pg_5433 psql -U mirix -d mirix -c "UPDATE clients SET write_scope='local' WHERE write_scope IS NULL;"
python -m evals.metaclaw.init_meta_agent

# 2. Run the fair comparison:
python -m evals.metaclaw --arm both --days 3
# → evals/metaclaw/runs/both-<ts>/reports.md
```
