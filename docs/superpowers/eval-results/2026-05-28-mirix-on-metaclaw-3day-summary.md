# MIRIX-on-MetaClaw 3-day Eval — Summary

> **See also**: `2026-05-29-mirix-vs-metaclaw-3day-fair-comparison.md` — the
> apples-to-apples both-arm run (metaclaw 34.85 % vs mirix 31.31 %, −3.5 pp,
> with MIRIX evolve fully healthy: 8/8 calls, 18 skills). This solo run (32.83 %)
> is a second healthy mirix data point; the two differ by run-to-run LLM variance.


**Run**: `evals/metaclaw/runs/mirix-20260528T101137Z`
**Date**: 2026-05-28
**Vendor**: aiming-lab/MetaClaw @ `fc163ba8a12ba5e6950544c3e0d55707c16e2a7e` (v0.4.1)
**Arm**: `mirix` (MIRIX is sole producer + storage of skills)
**Model**: `openai/gpt-5.2` via OpenRouter
**MIRIX server**: `http://127.0.0.1:8531` (branch `eval/original-mirix-3day`)

## Headline

| Metric | Value |
|---|---|
| Days run | 3 (day01–day03) |
| Total rounds | 33 |
| Correct (mean across n=3 retry) | 10.83 |
| **Overall accuracy** | **32.83 %** |
| Wall time | 29 m 23 s (1 763 s) |
| Estimated wallclock band | 33 m – 1 h 39 m |
| Total tokens | n/a (OpenRouter gpt-5.2 does not return usage) |

## Per-day breakdown

| Day | Questions | Correct | Accuracy | exact_match | f1 | iou |
|---|---|---|---|---|---|---|
| day01 | 10 | 3.0 | 30.0 % | 0.300 | 0.300 | 0.300 |
| day02 | 11 | 3.8 | 34.8 % | 0.273 | 0.354 | 0.346 |
| day03 | 12 | 4.0 | 33.3 % | 0.333 | 0.333 | 0.333 |

`passed` metric is 0.0000 across all days — MetaClaw's pass/fail oracle was not visible to the eval (by design, **narrow fairness**: both arms see only `{prompt_text, response_text}`, no pass/fail label leaks to the skill backend).

## MIRIX skill activity (this run only, fresh `user_id`)

| | Count |
|---|---|
| MirixSkillsAdapter `retrieve` calls | 96 |
| MirixEvolverAdapter `evolve` invocations | 8 |
| Skills MIRIX **created** | 18 |
| Skills MIRIX **edited** | 0 |
| Skills MIRIX **deleted** | 1 |
| Net new skills in MIRIX bank | 17 |
| Paper-side `add_skills` writes | **0** (suppressed by design — adapter returns `[]`) |

The 8 evolve calls correspond to the 8 multi-turn sessions paper segmented across day01–day03. Each evolve POSTs the session transcript tail to `MIRIX /v1/skills/evolve`, which lets MIRIX's `ProceduralMemoryAgent` decide what skills to create/edit/delete server-side.

Retrieve calls average ~3 per round (queries fire at user_prompt, post-tool, and reflective steps inside the agent loop). Each returns `top-k=6` skills, prepended to the agent context.

## Day01 comparison vs `metaclaw` arm (from earlier `--arm both --days 1`)

| Arm | day01 accuracy |
|---|---|
| metaclaw (paper's skills_dir backend) | 30.0 % |
| mirix (this work) | 40.0 % (single run) → 30.0 % (this 3-day run, day01 only) |

The single 1-day mirix run gave 40 %; this 3-day run gave 30 % on day01 with a different fresh `user_id`. Variance across runs is high at 10-question granularity — a single-day delta is not statistically meaningful with n=3 retry. The 3-day mean (32.83 %) is the better signal for this configuration.

## Architecture verified working in this run

- **MIRIX produces and stores skills**, not paper. Paper local skills tempdir was empty after the run.
- **D6 dispatch** (`METACLAW_SKILLS_PROVIDER=mirix`, `METACLAW_EVOLVER_PROVIDER=mirix`) cleanly swaps paper's defaults for MIRIX-backed adapters; total D6 mod surface in vendored `launcher.py` ≤ 100 lines (all annotated `# [D6 mod 2026-05-28]`).
- **clawdbot pre-warm** kept gateway startup under the launcher's 15 s internal `openclaw config set` timeout on a previously-warm openclaw; cold start succeeded after 2× `openclaw config get` warm-up.
- **fairness invariant**: same `all_tests_metaclaw.json` is sliced to `days=3` once and shared across arms when `--arm both` is used.

## Caveats

1. **Token columns are `n/a`**. OpenRouter does not relay `usage` for `openai/gpt-5.2` in a shape paper's bench parses. Accuracy is unaffected. To get token attribution, run against OpenAI direct or pin to a model OpenRouter does relay usage for.
2. **Single seed, no error bars**. n=3 retry is per-question, but the dataset itself is one ordering. Repeat the run with different seeds to get accuracy confidence intervals.
3. **`passed` metric is 0**. MetaClaw's `passed` requires the oracle that we explicitly do not pipe to either skill backend (fairness). All accuracy here is the per-field exact-match / f1 / iou family, identical to what paper reports for its own `skills_only` mode.

## Reproducing

```bash
# from repo root
export OPENROUTER_API_KEY=<your_key>   # in .env already
python scripts/start_server.py --port 8531   # MIRIX server (branch with /v1/skills/evolve)
python -m evals.metaclaw --arm mirix --days 3 --yes
# results land in: evals/metaclaw/runs/mirix-<utc-ts>/
```

To get an apples-to-apples baseline-vs-mirix comparison in one shot:

```bash
python -m evals.metaclaw --arm both --days 3 --yes
# produces reports.md with metaclaw-arm + mirix-arm side-by-side
```
