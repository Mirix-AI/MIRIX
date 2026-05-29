# MetaClaw 3-day eval — Original-MIRIX (pre-skill-evolve) arm

**Run id**: `20260512-174714-3ce960`
**Date**: 2026-05-12
**Branch**: `eval/original-mirix-3day` (from `main` HEAD `d6e7c14`)
**Arm**: `legacy` (original MIRIX procedural memory: `summary+steps` schema, old `procedural_memory_insert/update` tools)
**Model**: OpenRouter `openai/gpt-5.2` (chat) + `google/gemini-embedding-001` (embedding, dim 1536)
**Bench**: `metaclaw-bench/eval/day01..day03` (P1 ISO 8601 preference arc)
**Grain**: 1 evolve message per round + 1 retrieve per round
**Postgres**: `mirix_pgvector_legacy_eval` (port 5434)
**MIRIX server**: port 8532
**Reflection prefix (D10)**: not applied (R1 tripwire stayed clear)
**Wallclock**: ~50 min (2026-05-12T17:47:14 → T18:37:41)

## Per-day accuracy

| Day | Rounds | Correct | Accuracy |
|---|---|---|---|
| day01 | 10 | 4 | 0.40 |
| day02 | 11 | 5 | 0.45 |
| day03 | 12 | 9 | 0.75 |
| **mean** | 33 | 18 | **0.545** |

## Per-round outcome details

### day01 (4/10 = 0.40)

| Round | Type | Outcome | Reward |
|---|---|---|---|
| r1 | file_check | pass | 1.0 |
| r2 | file_check | fail | 0.0 |
| r3 | multi_choice | pass | 1.0 |
| r4 | file_check | pass | 1.0 |
| r5 | file_check | fail | 0.0 |
| r6 | multi_choice | fail | 0.0 |
| r7 | file_check | fail | 0.0 |
| r8 | multi_choice | fail | 0.0 |
| r9 | file_check | pass | 1.0 |
| r10 | multi_choice | fail | 0.0 |

### day02 (5/11 = 0.45)

| Round | Type | Outcome | Reward |
|---|---|---|---|
| r1 | file_check | pass | 1.0 |
| r2 | file_check | pass | 1.0 |
| r3 | file_check | fail | 0.0 |
| r4 | multi_choice | pass | 1.0 |
| r5 | file_check | fail | 0.0 |
| r6 | file_check | fail | 0.0 |
| r7 | multi_choice | pass | 1.0 |
| r8 | file_check | fail | 0.0 |
| r9 | multi_choice | fail | 0.0 |
| r10 | file_check | fail | 0.0 |
| r11 | multi_choice | pass | 1.0 |

### day03 (9/12 = 0.75)

| Round | Type | Outcome | Reward |
|---|---|---|---|
| r1 | file_check | pass | 1.0 |
| r2 | file_check | fail | 0.0 |
| r3 | multi_choice | pass | 1.0 |
| r4 | file_check | pass | 1.0 |
| r5 | file_check | fail | 0.0 |
| r6 | multi_choice | pass | 1.0 |
| r7 | file_check | pass | 1.0 |
| r8 | file_check | pass | 1.0 |
| r9 | multi_choice | pass | 1.0 |
| r10 | file_check | fail | 0.0 |
| r11 | multi_choice | pass | 1.0 |
| r12 | file_check | pass | 1.0 |

## Procedural memory growth

End-of-run totals from `mirix_pgvector_legacy_eval`:

| entry_type | count |
|---|---|
| workflow | 17 |
| guide | 2 |
| **total** | **19** |

The procedural_memory_agent stored entries on every day (R1 tripwire stayed clear throughout); no D10 reflection prefix was needed.

## Notes

- **Bootstrap requirements (not anticipated by R6 in the spec)**: the seeded default client has `write_scope=null, read_scopes=[]`. Without `write_scope`, `/memory/add_sync` returns 403. Without matching `read_scopes`, `/memory/search` returns 0 rows even when writes are durable — this is the silent failure mode that would have collapsed the arm to a no-skills baseline. The working bootstrap is:
  1. `POST /clients/create_or_get` (idempotent — server auto-seeds on first boot)
  2. `PATCH /clients/<id>` with body `{id: <id>, write_scope: "eval-legacy", read_scopes: ["eval-legacy"]}`
  3. `POST /agents/meta/initialize` with full LLM/embedding config

- The old procedural_memory_agent showed a strong **bias toward `workflow` entries** (17/19 = 89%), with only 2 `guide` entries. This is the pre-skill-evolve agent's default classification.

- Compared to the feat/skill-evolve arm (mean 0.56 from `2026-05-08-metaclaw-3day-summary.md`), the legacy arm's mean is essentially tied (0.545 vs 0.56) but the day-by-day curves differ substantially. See `2026-05-12-metaclaw-3day-three-way-comparison.md` for the side-by-side.

## Files

- Source: `evals/metaclaw/reports/20260512-174714-3ce960/{summary.md,summary.json,day01_metrics.json,day02_metrics.json,day03_metrics.json,workspace/}`
- Archived JSON: `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.json`
