# MetaClaw paper-aligned gating — Four-arm comparison

**Date**: 2026-05-20
**Bench**: `metaclaw-bench-small` first 3 test(s)
**Phase**: gating
**Harness**: paper original (openclaw subprocess + metaclaw-bench infer)
**Model**: `openai/gpt-5.2` (PRD D10 — paper's published model)

## Arms

| Arm | Description | Status | Output |
|---|---|---|---|
| A (armA-mirix-skills) | paper proxy + MIRIX-skill-evolve backend (:8531) | ok | `evals/metaclaw_aligned/runs/armA-mirix-skills-small-20260520-160528/infer/run_20260520_160547/report.json` |
| B (armB-mirix-legacy) | paper proxy + MIRIX-legacy backend (:8532) | ok | `evals/metaclaw_aligned/runs/armB-mirix-legacy-small-20260520-160531/infer/run_20260520_160545/report.json` |
| C (armC-baseline) | paper baseline (no proxy, agent → LLM direct) | ok | `evals/metaclaw_aligned/runs/armC-baseline-small-20260520-160533/infer/run_20260520_160533/report.json` |
| D (armD-paper-native) | paper proxy + paper-native SkillManager (anchor) | ok | `evals/metaclaw_aligned/runs/armD-paper-native-small-20260520-160535/infer/run_20260520_160551/report.json` |

## Per-test accuracy

| Test | A | B | C | D |
|---|---|---|---|---|
| day01 | 0.40 (2/5) | 0.40 (2/5) | 0.40 (2/5) | 0.40 (2/5) |
| day02 | 0.37 (1/5) | 0.40 (2/5) | 0.40 (2/5) | 0.37 (1/5) |
| day03 | 0.27 (1/5) | 0.30 (1/5) | 0.23 (1/5) | 0.20 (1/5) |
| **mean** | **0.344** (5/15) | **0.367** (6/15) | **0.344** (5/15) | **0.322** (5/15) |

## Pairwise deltas (mean)

| Pair | Δ |
|---|---|
| A − B | -0.022 |
| A − C | +0.000 |
| A − D | +0.022 |
| B − C | +0.022 |
| B − D | +0.044 |
| C − D | +0.022 |

## Known deviations from paper

1. **Skill backend** (arms A, B only): MIRIX serves the skill backend in arms A and B, where paper uses its own `metaclaw/skill_manager.SkillManager`. arms C and D are paper original.
2. **Single seed**: no confidence intervals; all numbers are point estimates.
3. **Mode coverage**: only `skills_only` and `baseline` modes. paper's other modes (memory_run, buffer_memory_run, madmax_memory_run, rl_*, proxy_passthrough_run) are out of scope (PRD).
4. **Subset**: paper-small first 3 of 12 tests (gating phase per PRD Q5.b option G1). Main run will use full 30 tests.

## Run metadata

| Arm | Started | Finished | Wall sec | exit | MIRIX URL | MIRIX user_id |
|---|---|---|---|---|---|---|
| A | 20260520-160528 | 20260520-162422 | 1115.7 | 0 | http://127.0.0.1:8531 | eval-metaclaw-aligned-gating |
| B | 20260520-160531 | 20260520-164004 | 2059.3 | 0 | http://127.0.0.1:8532 | eval-metaclaw-aligned-gating |
| C | 20260520-160533 | 20260520-162205 | 992.0 | 0 | — | — |
| D | 20260520-160535 | 20260520-162654 | 1263.2 | 0 | — | — |

