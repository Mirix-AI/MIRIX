# MetaClaw 3-day eval — Three-way comparison

**Date**: 2026-05-12
**Bench**: `metaclaw-bench/eval/day01..day03` (P1 ISO 8601 preference arc)
**Model**: OpenRouter `openai/gpt-5.2` (chat) + `google/gemini-embedding-001` (embedding, dim 1536)
**Agent loop / scorer**: identical across all three arms (`evals/metaclaw/round_runner.py`)
**Grain**: 1 evolve message per round + 1 retrieve per round (identical across arms A & B; no evolve/retrieve in arm C)

## Arms

| Arm | Branch | Procedural memory | Evolve endpoint | Retrieve endpoint | Wallclock |
|---|---|---|---|---|---|
| **A. feat/skill-evolve** | `feat/skill-evolve` | `name/description/instructions/triggers/examples/version` schema, 5 CLI tools | `POST /v1/skills/evolve` | `GET /v1/skills?query=...` | ~25 min (13:44 → 14:09) |
| **B. original-MIRIX** | `eval/original-mirix-3day` (from `main`) | `summary/steps/entry_type` schema, old `procedural_memory_insert/update` tools | `POST /memory/add_sync` | `GET /memory/search?memory_type=procedural` | ~50 min (17:47 → 18:37) |
| **C. no-skills** | `feat/skill-evolve --no-skills` | (none — cold every round) | (none) | (none) | ~20 min (15:19 → 15:40) |

## Per-day accuracy

| Day | Rounds | A. feat/skill-evolve | B. original-MIRIX | C. no-skills | Δ (A − C) | Δ (B − C) | Δ (A − B) |
|---|---|---|---|---|---|---|---|
| day01 | 10 | 0.10 (1/10) | 0.40 (4/10) | 0.20 (2/10) | −0.10 | +0.20 | −0.30 |
| day02 | 11 | 0.91 (10/11) | 0.45 (5/11) | 0.27 (3/11) | +0.64 | +0.18 | +0.46 |
| day03 | 12 | 0.67 (8/12) | 0.75 (9/12) | 0.50 (6/12) | +0.17 | +0.25 | −0.08 |
| **mean** | 33 | **0.576** (19/33) | **0.545** (18/33) | **0.333** (11/33) | **+0.242** | **+0.212** | **+0.030** |

Source values pulled from the actual JSON/MD summaries on disk (see "Source files" below). Means here are computed as `total_passed / total_rounds` over the 33 rounds; the simple per-day average is very close (A=0.560, B=0.533, C=0.323) but slightly different from the round-weighted mean because day01–03 have unequal round counts (10, 11, 12).

## Interpretation

**A vs C (skill-evolve vs no-skills): +0.24 mean lift.** The new procedural memory stack delivers meaningful improvement over no memory at all, but the lift is **concentrated almost entirely on day02** (+0.64 on a single day), with day01 actually slightly worse than baseline (−0.10) — likely cold-start LLM noise on a fresh skill bank where the evolve step hadn't yet produced anything useful to retrieve against.

**B vs C (original-MIRIX vs no-skills): +0.21 mean lift.** The pre-skill-evolve procedural memory also delivers meaningful lift over no memory — close in magnitude to arm A's net lift, but with a notably different shape. The lift is **more evenly distributed** across days (+0.20 / +0.18 / +0.25), suggesting the legacy `procedural_memory_agent` warms up faster but has a lower ceiling.

**A vs B (skill-evolve vs original-MIRIX) — the headline experiment**: **+0.03 mean** (essentially zero), with strongly **divergent per-day curves**:
- day01: arm A is **−0.30 worse** than arm B (cold start hurt arm A more — its evolve step produced exactly 1 skill on day01, while arm B's prolific legacy agent had already laid down several rows that started paying off mid-day)
- day02: arm A is **+0.46 better** (after one day of accumulated skills, arm A's structured `triggers/examples` schema started paying off, and a single high-quality skill — `normalize-project-management-json-time-fields-iso8601-plus0800` — appears to have been retrieved on most of the 10 passing rounds)
- day03: arm A is **−0.08 worse** (arm B's accumulated 17+ workflow rows held up; arm A's 3-skill bank may have been retrieved-but-not-quite-applicable to day03's task mix)

The headline number is that **the two arms tied on aggregate accuracy** (A=0.576 vs B=0.545; +0.03 is well within single-seed noise) — the skill-evolve refactor did not on its own change the bottom-line score, but it dramatically reshuffled when the lift arrives. Arm A's pattern (catastrophic day01, blockbuster day02, mid day03) suggests the new schema has higher variance: bigger payoffs once warmed up, but also bigger cold-start cost and possibly poorer generalization across task domains.

## Variables held constant (control)

- Bench data (`metaclaw-bench/eval/day01..day03`)
- Qwen3-style tool loop (`evals/metaclaw/round_runner.py`)
- LLM (`openai/gpt-5.2` via OpenRouter)
- Embedding (`google/gemini-embedding-001` via OpenRouter)
- Scorers (multi_choice regex + file_check subprocess)
- Cold start (no preloaded procedures)
- Workspace persistence (shared across all 3 days within a run)

## Variables that differ (causal axis)

- **Procedural memory ORM schema**: `name/description/instructions/triggers/examples/version` (arm A) vs `summary/steps/entry_type` (arm B)
- **procedural_memory_agent prompt**: skill-evolve CLI-style vs. legacy insert/update
- **procedural_memory tools**: 5 CLI tools (`skill_list/read/create/edit/delete`) vs. 2 (`procedural_memory_insert/update`)
- **REST endpoints**: `/v1/skills/*` (arm A) vs unprefixed `/memory/add_sync` + `/memory/search` (arm B)
- **Postgres instance**: port 5433 (arm A) vs port 5434 (arm B), independent volumes

## Caveats

- **Single seed.** Confidence intervals not estimated — would require ≥3 runs per arm. Arm A's day01=0.10 vs day02=0.91 swing is large enough that it could simply be one bad seed; the day02 result alone dominates arm A's mean.
- **Day01 is cold-start for arms A & B**; numbers there are sensitive to LLM noise more than to memory differences. The −0.30 A−B gap on day01 is the noisiest single comparison in the table.
- **R6 bootstrap gap** (documented in `2026-05-12-metaclaw-3day-legacy-original-mirix-summary.md`): the default-seeded MIRIX client has `write_scope=null, read_scopes=[]`. Without PATCHing both to match, arm B would have silently degraded into the no-skills baseline (writes 403, reads return 0 rows). The first smoke run caught this; subsequent runs used the corrected bootstrap.
- **Procedural growth shape differs**: arm A produced 3 named skills across the 3 days; arm B produced 19 procedural-memory rows (17 workflow + 2 guide). The legacy `procedural_memory_agent` is much more prolific (~6× more rows), which may partially explain the more uniform day-by-day lift in arm B and arm A's high-variance pattern.
- **Wallclock asymmetry**: arm B took ~2× longer than arm A (~50 min vs ~25 min), likely because the legacy agent emits more tool calls per evolve. Not a fairness issue (accuracy is the metric, not throughput), but worth noting for any future cost analysis.

## Source files

- Arm A summary: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.{md,json}`
- Arm B summary: `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.{md,json}`
- Arm C summary: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.{md,json}`
- Prior 2-way (A vs C) comparison: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-comparison.md`
