# MIRIX × MetaClaw Evolution-Bench Eval Harness

Evaluates MIRIX's procedural memory subsystem as a drop-in replacement
for MetaClaw's `SkillEvolver` / `SkillManager` on `metaclaw-bench`
days 01–03 (P1 ISO 8601 datetime preference).

See `docs/superpowers/specs/2026-05-08-mirix-metaclaw-eval-design.md`
for the design.

## Prerequisites

1. PostgreSQL container running (`docker ps | grep e2e-postgres`).
2. MetaClaw cloned to `third_party/MetaClaw/` (see plan Task 1).
3. Environment variables exported (see `.env.example` below).

## Environment

```bash
export OPENAI_API_KEY="<your-openrouter-key>"        # OpenRouter key
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export EVAL_CHAT_MODEL="openai/gpt-5.2"
export EVAL_EMBED_MODEL="google/gemini-embedding-001"
export EVAL_EMBED_DIM="1536"                          # truncate gemini to MIRIX dim
```

## Running

Start the MIRIX API server in a separate terminal:

```bash
python scripts/start_server.py --port 8531
```

Then run the harness:

```bash
# Smoke (one round)
python -m evals.metaclaw.run_3day_eval --days day01 --max-rounds 1

# Full 3-day e2e
python -m evals.metaclaw.run_3day_eval
```

Reports land in `evals/metaclaw/reports/<run-id>/`.
