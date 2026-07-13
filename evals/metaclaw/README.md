# MetaClaw 30-day eval

This package runs the vendored MetaClaw 30-day benchmark under two arms:

- **`metaclaw`** — upstream skill backend (file-based skill bank under the proxy)
- **`mirix`** — MIRIX REST-backed skill retrieval + evolution

See `cli.py` for the CLI entry point (`python -m evals.metaclaw …`) and
`runner.py` for the per-arm orchestration.

## Reproducing a run

On this branch the MIRIX core (`mirix/`, `tests/`) is byte-identical to the
`feat/procedural-memory-main-pr` production branch, so benchmark results
exercise exactly the skill system proposed for `main`. The harness talks to
MIRIX only through the generic memory REST API; the wire contract is pinned by
`tests/test_server_contract.py` in this package.

1. `pip install -e ".[eval]"` (adds fastapi/uvicorn/httpx/click/pyyaml/etc.).
2. **Dataset (not in git):** `evals/metaclaw/data/` (~38 MB — `all_tests.json`,
   `eval/day01..day30/`, `openclaw_cfg/`, `workspaces/`) is gitignored by the
   broad `data/` rule. Copy it out-of-band from a machine that has it, or
   re-vendor it from upstream MetaClaw at the exact pin recorded in
   `evals/metaclaw/METACLAW_VERSION`.
3. Start MIRIX (needs Postgres + Redis, see the repo root README):
   `python scripts/start_server.py --port 8531`.
4. Put `OPENROUTER_API_KEY=…` in the repo-root `.env` (or export
   `BENCHMARK_API_KEY`), then run e.g.
   `python -m evals.metaclaw --arm mirix-generic --days 3 --yes`.

Run artifacts land under `evals/metaclaw/runs/` (gitignored).

## Smoke tests (offline)

Verifies the eval plumbing without spending real LLM tokens, without spawning
the real `clawdbot` / `openclaw` daemons, and without requiring a running MIRIX
server. Uses tiny FastAPI stubs (LLM-shaped + MIRIX-shaped) injected via the
runner's DI hooks (`proxy_starter`, `proxy_stopper`, `bench_runner`,
`extra_env`).

    pytest -m integration evals/metaclaw/tests/test_smoke.py -v

Expected runtime: well under 2 min. Default pytest runs (without
`-m integration`) skip these tests via the project's `pytest.ini`.

These smoke tests are safe to run concurrently with a live eval — they never
invoke `clawdbot`, `openclaw`, the vendored bench subprocess, or the real
MetaClaw proxy.
