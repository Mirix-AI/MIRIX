# MetaClaw 30-day eval

This package runs the vendored MetaClaw 30-day benchmark under two arms:

- **`metaclaw`** — upstream skill backend (file-based skill bank under the proxy)
- **`mirix`** — MIRIX REST-backed skill retrieval + evolution

See `cli.py` for the CLI entry point (`python -m evals.metaclaw …`) and
`runner.py` for the per-arm orchestration.

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
