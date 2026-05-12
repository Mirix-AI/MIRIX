# Original-MIRIX 3-Day MetaClaw Eval — Design Spec

**Date**: 2026-05-12
**Author**: nanjiayan
**Status**: approved (verbal)
**Companion plan**: `docs/superpowers/plans/2026-05-12-original-mirix-3day.md` (to be written)
**Related runs**:
  - feat/skill-evolve: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.{md,json}`
  - no-MIRIX baseline: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.{md,json}`
  - feat vs baseline: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-comparison.md`

---

## 1. Goal

Produce a third evaluation arm: **MIRIX with its pre-skill-evolve procedural memory** (main HEAD `d6e7c14`) on MetaClaw bench `day01..day03` of the P1 ISO 8601 preference arc, under the **same** agent loop, model, scorer, dataset and grain as the existing feat/skill-evolve and no-MIRIX runs.

Output: a clean three-way A/B/C table that isolates **what the new skill-based procedural-memory schema + agent prompt + tools contribute on top of the original MIRIX procedural memory**, with the agent loop and bench held constant.

## 2. Locked Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | Mirror feat/skill-evolve grain exactly: 1 evolve message per round + 1 retrieve per round | One free variable: schema/prompt/tools. Day-grain reflection would add a second variable. |
| D2 | New branch `eval/original-mirix-3day` cut from `main` (d6e7c14); independent postgres container `mirix_pgvector_legacy_eval` on port 5434 | Main MIRIX code is unpolluted; db schema cannot leak from feat/skill-evolve. |
| D3 | Reuse `round_runner.py`, `llm_config_helpers.py`, most of `run_3day_eval.py`, and `format_adapter.py` verbatim via cherry-pick from feat/skill-evolve | Maximises code-path overlap → maximises comparability. |
| D4 | Three new files under `evals/metaclaw/`: `mirix_legacy_client.py`, `mirix_legacy_evolver.py`, `mirix_legacy_manager.py`. Adapters that hide the schema gap from `round_runner`. | Adapter pattern keeps consumer code zero-change. |
| D5 | `format_adapter.legacy_procedural_to_metaclaw(row)` returns the **same dict shape** (`name/description/content/category`) as `mirix_to_metaclaw` | System-prompt injection template stays identical. |
| D6 | Legacy server runs on `--port 8532`; the `--legacy` CLI flag of `run_3day_eval.py` defaults `--mirix-url` to `http://127.0.0.1:8532` | Avoids port collision with any running feat/skill-evolve server (8531). |
| D7 | LLM stack identical to other runs: OpenRouter chat `openai/gpt-5.2`, embedding `google/gemini-embedding-001` | Model held constant across all three arms. |
| D8 | Cold start (no preloaded procedures); workspace persists across the 3 days | Same convention as feat/skill-evolve run. |
| D9 | No day-boundary reflection message — every round flushes one evolve message and that's it | Strict grain match with feat/skill-evolve. |
| D10 | If procedural_memory_agent never writes (R1), add a reflection prefix to evolve messages: `"Reflect on the round below and extract any reusable procedural knowledge:"`. Fallback only — not default. | Detect rather than assume. |

## 3. Architecture

```
branch: eval/original-mirix-3day  (from main d6e7c14)

  ┌──────────────────────┐         ┌────────────────────────────────┐
  │ Qwen3 tool-call loop │         │ MIRIX server (main HEAD)        │
  │ round_runner.py      │  evolve │  • meta_agent                   │
  │ (unchanged)          ├────────▶│    └─ procedural_memory_agent   │
  │                      │ POST    │       (old prompt + old tools)  │
  │                      │ /memory │  • ORM: summary+steps           │
  │                      │ /add_   │  • postgres                     │
  │                      │ sync    │    mirix_pgvector_legacy_eval   │
  │                      │         │    (port 5434)                  │
  │                      │ retrieve│                                 │
  │                      ├────────▶│  GET /memory/search?         │
  │                      │ system  │       memory_type=procedural    │
  │                      │ prompt  │       &query=...                │
  └──────────────────────┘         └────────────────────────────────┘
            │
            ▼
  reports/<run-id>/summary.{json,md}
  docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix.*
```

**Invariants (control variables)**: bench data (`metaclaw-bench/eval/day01..day03`), Qwen3 tool loop, LLM, embedding, scorers, evolve/retrieve grain, cold start, workspace persistence.

**Variables under test (causal)**: procedural memory ORM schema (`summary+steps` ↔ `name/description/instructions/triggers/examples/version`), procedural_memory_agent prompt and tools, REST endpoints (legacy `/memory/*` unprefixed on main ↔ skill-evolve `/v1/skills/*` literal).

## 4. Components & Contracts

### 4.1 `evals/metaclaw/mirix_legacy_client.py` (new)

```python
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_META_AGENT_NAME = "meta_memory_agent"

class LegacyMirixClient:
    def __init__(self, base_url="http://127.0.0.1:8532", user_id="eval-legacy-3day",
                 client_id=DEFAULT_CLIENT_ID, timeout=600.0, _client=None): ...

    async def _resolve_meta_agent_id(self) -> str:
        """One-time bootstrap: GET /agents → find the agent whose name is
        DEFAULT_META_AGENT_NAME. Cached on the instance after first call.
        Raises if no meta agent exists (caller can call /agents/meta/initialize)."""

    async def add_memory(self, message_text: str) -> dict:
        """POST /memory/add_sync. Synchronous: blocks until the meta_agent
        finishes processing (which is exactly the behaviour we want — by the
        time evolve returns, procedural_memory writes are durable).

        Wire body (AddMemoryRequest):
            {
              "meta_agent_id": <resolved id>,
              "messages": [{"role": "user", "content": message_text}],
              "user_id": self.user_id,
              "chaining": True
            }
        """

    async def search_procedural(self, query: str, limit: int = 6) -> list[dict]:
        """GET /memory/search with query params:
            memory_type=procedural
            search_method=bm25            # explicit — server default is 'embedding'
            search_field=summary          # primary descriptive field on old schema
            query=<query>
            limit=<limit>
            user_id=<user_id>
        Returns rows shaped {summary, steps, entry_type, ...}.

        Note: server's main-branch handler also auto-searches the 'steps' field
        when search_field is omitted, but explicit 'summary' is chosen here so
        BM25 ranks against the human-readable description (matching the spirit
        of feat/skill-evolve's 'description'-field hard-coded search)."""

    async def health(self) -> bool: ...
    async def aclose(self): ...
```

**Auth**: `X-Client-Id` header on every request, defaulting to the seeded admin client.

**Why `add_sync` not `add`**: `/memory/add` is Kafka-backed (`put_messages`) and returns `{status: "queued"}` immediately. The next round's retrieve might miss writes from the previous round → silent confounder. `/memory/add_sync` calls `server.send_messages(...)` inline and blocks until the agent finishes (`rest_api.py` lines 2131-2236 on main). Default path is `add_sync`; `add` is not used.

**meta_agent_id bootstrap**: on first `add_memory`/`search_procedural` call (or via an explicit `await client.ensure_ready()` from the driver), the client fetches `GET /agents` and picks the agent whose name is `meta_memory_agent`. The resulting id is cached. If the row is missing, the driver must call `POST /agents/meta/initialize` once at server-bootstrap time (handled in §8 step 2).

### 4.2 `evals/metaclaw/mirix_legacy_evolver.py` (new)

```python
class LegacyMirixEvolver(SkillEvolver):
    def __init__(self, mirix: LegacyMirixClient):
        # bypass parent __init__ — we only need interface parity.
        # update_history / history_path kept as no-op state because
        # inherited get_update_summary() reads update_history.
        self.mirix = mirix
        self.update_history: list[dict] = []
        self.history_path = None

    def should_evolve(self, batch, threshold: float = 0.0) -> bool:
        # Driver-driven: always allow.
        return True

    async def evolve(
        self,
        failed_samples: Iterable[RoundResult],
        current_skills: dict | None = None,   # signature parity
    ) -> list[dict]:
        """Iterate rounds, POST each to /memory/add_sync. Per-round errors
        are logged and skipped (one bad POST must not nuke a 12-round day —
        see §6 'POST /memory/add non-2xx'). Returns [] because legacy
        /memory/add_sync does not surface the created procedural_memory
        rows in its response, and querying for the diff would be racy."""
        for r in failed_samples:
            try:
                await self.mirix.add_memory(round_to_message(r))
            except Exception as e:
                logger.warning("legacy evolve POST failed for round %s: %s", r.round_id, e)
        return []
```

Mirrors `MirixSkillEvolver` exactly — same parent class, same signature `(failed_samples, current_skills=None) -> list[dict]`, same no-op `update_history`/`history_path` fields, same `should_evolve(batch, threshold)` signature. The only meaningful difference is that the result list is always empty (because main's `/memory/add_sync` doesn't return created rows, and we choose not to make a second roundtrip).

### 4.3 `evals/metaclaw/mirix_legacy_manager.py` (new)

```python
DEFAULT_TOP_K = 6

class LegacyMirixManager(SkillManager):
    def __init__(self, mirix: LegacyMirixClient):
        self.mirix = mirix

    async def retrieve_async(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        rows = await self.mirix.search_procedural(query=query, limit=top_k)
        return [legacy_procedural_to_metaclaw(r) for r in rows]
```

### 4.4 `evals/metaclaw/format_adapter.py` (extend)

Add one function:

```python
def legacy_procedural_to_metaclaw(row: dict) -> dict:
    """Map old procedural_memory row → MetaClaw skill-shape."""
    return {
        "name":        row.get("entry_type") or "procedure",
        "description": row.get("summary") or "",
        "content":     row.get("steps") or "",
        "category":    row.get("entry_type") or "procedure",
    }
```

### 4.5 `evals/metaclaw/run_3day_eval.py` (small patch)

Add `--legacy` flag (mutually exclusive with `--no-skills`). Branches the MIRIX wiring:

```python
if args.legacy:
    mirix = LegacyMirixClient(base_url=args.mirix_url, user_id=args.user_id, timeout=args.mirix_timeout)
    evolver = LegacyMirixEvolver(mirix=mirix)
    skill_mgr = LegacyMirixManager(mirix=mirix)
elif use_mirix:
    mirix = MirixClient(...)
    evolver = MirixSkillEvolver(mirix=mirix)
    skill_mgr = MirixSkillManager(mirix=mirix)
```

When `--legacy`, default `--mirix-url` to `http://127.0.0.1:8532` if user didn't override.

## 5. Data Flow (one round)

```
Round N starts
  │
  ├─ skill_mgr.retrieve_async(query=question, top_k=6)
  │     GET /memory/search?memory_type=procedural&query=<question>&limit=6
  │       → procedural_memory_manager.list_procedures (BM25 on summary+steps)
  │       → rows
  │     legacy_procedural_to_metaclaw(rows) → [{name, description, content, category}]
  │
  ├─ run_round(question, retrieved_skills, ...)
  │     system prompt with retrieved skills (identical template)
  │     Qwen3 tool loop with bash / read_file / write_file / list_dir
  │     boxed answer | file at workspace/.../answer.txt
  │
  ├─ score_multi_choice | score_file_check → RoundResult
  │
  └─ evolver.evolve_async([RoundResult])
        for r in rounds:
            POST /memory/add  body={"message": round_to_message(r), "user_id": ...}
              → meta_agent → procedural_memory_agent
              → decides whether to call procedural_memory_insert/update
              → row appended to procedural_memory table

Day N ends → same workspace dir reused for Day N+1
3 days end → reports/<run-id>/summary.{json,md}
```

## 6. Error Handling

| Failure | Handling |
|---------|----------|
| `POST /memory/add` non-2xx | Log full response body + status; record `evolve_error` on the RoundResult; continue to next round. Do not raise (evolve failure must not kill a 12-round run). |
| `GET /memory/search` non-2xx | Treat as empty retrieval; record `retrieve_error`; continue. |
| `httpx.HTTPError` (network / timeout) | Same as non-2xx — log and degrade to empty skills. |
| `procedural_memory_agent` writes 0 rows after first day | Tripwire (see §7 R1). |
| Server not reachable on port 8532 at startup | Driver fails fast with a clear message ("`Is the legacy server running? Try \`python scripts/start_server.py --port 8532\``"). |
| Bash sandbox / file_check unchanged from feat/skill-evolve run | Same hardening already in `round_runner.py`. |

## 7. Risks & Mitigations

| ID | Risk | Detection | Mitigation |
|----|------|-----------|------------|
| R1 | procedural_memory_agent's old prompt is conservative and never calls `procedural_memory_insert/update` for our evolve messages → retrieve always returns empty → arm collapses to no-skills baseline. | After day01, `SELECT count(*) FROM procedural_memory WHERE user_id='eval-legacy-3day'`. If 0, R1 fired. | Apply D10: add the reflection prefix to evolve messages and re-run. Document the toggle in the result write-up. |
| R2 | `/memory/add` is queue-backed (Kafka) so the write is async; the next round's retrieve may not see it yet. | (Pre-emptively avoided.) | **Default path** is `/memory/add_sync` (line 2131 on main), which calls `send_messages` inline. `/memory/add` is never used by this harness. |
| R3 | DB migration drift if the legacy postgres is somehow reused. | Hard-isolate: dedicated container, dedicated volume, port 5434. Server is started against `MIRIX_PG_URI` pointing at port 5434. | Don't share databases. Bring up via separate docker-compose file or `docker run`. |
| R4 | Port 8531 already taken by the feat/skill-evolve server on a dev machine. | Driver health-check before run. | Legacy server uses 8532; driver defaults mirror that when `--legacy`. |
| R5 | The structured markdown of `round_to_message` is not interpreted as a "task to reflect on" by the old meta_agent → routing falls to the chat path. | Same tripwire as R1. | Same mitigation as R1 (D10 prefix). |
| R6 | Main may not seed the default admin client row. | `GET /clients` after server start. | Server-start subroutine calls `POST /clients/create_or_get` against `client-00000000-0000-4000-8000-000000000000`. |
| R7 | Old procedural_memory schema requires `entry_type` (NOT NULL?). If the agent fails to set it, writes raise. | Watch server logs during dry-run. | Read main's ORM and schema — if `entry_type` is required, fallback prompt nudges the agent to set it. |
| R8 | Embedding column shape mismatch — main may not write embeddings for procedural_memory. | Inspect rows after first write. | Acceptable. BM25 search path uses text columns, not embeddings, so retrieval still works. |
| R9 | `meta_memory_agent` row does not exist after fresh server boot, so `_resolve_meta_agent_id` raises. | Driver pre-flight: `GET /agents`. | Driver calls `POST /agents/meta/initialize` once before the first day. This is the same bootstrap step the dashboard performs on first launch. |

## 8. Testing & Verification

**Pre-run smoke (sequential gates)**

1. `pytest -q evals/metaclaw/tests/` — all existing tests pass.
2. Spin up legacy postgres on 5434 + legacy server on 8532. `GET /health` returns 200. **Bootstrap**: `POST /clients/create_or_get` (default admin client) → `POST /agents/meta/initialize` (meta_memory_agent + 6 children) → `GET /agents` to verify `meta_memory_agent` exists.
3. **Dry-run round** with `--legacy --dry-run` — driver path through new client / evolver / manager runs without exception.
4. **Single live round** with `--legacy --max-rounds 1` on day01 — verify exactly one row appears in `procedural_memory` table for the legacy user.
5. **R1 tripwire**: if step 4 produces 0 rows, switch on the reflection prefix and re-test before committing to the full 3-day run.

**Full run**

6. `python -m evals.metaclaw.run_3day_eval --legacy --days day01,day02,day03 --max-rounds 12 --top-k 6 --user-id eval-legacy-3day --mirix-url http://127.0.0.1:8532` (≈ 25-45 min runtime).
7. Produce `summary.{json,md}` + `day_metrics.json` + 3 archived skill snapshots.

**Cross-run comparison**

8. Write `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-three-way-comparison.md` — table of mean accuracy by day across feat/skill-evolve / original-mirix / no-skills.

## 9. Out of Scope

- N-seed runs for confidence intervals (single seed first; CI work is a follow-up).
- Day04+ extension (P2 file_naming arc).
- Modifying `mirix/` on `eval/original-mirix-3day` — main MIRIX code is read-only here.
- Re-running the feat/skill-evolve arm — its numbers stand and are referenced by file.
- Original paper's `metaclaw.skill_evolver.SkillEvolver` as a fourth arm.

## 10. Deliverables

| Artifact | Location |
|----------|----------|
| Branch | `eval/original-mirix-3day` |
| Docker isolation | `docker run -d --name mirix_pgvector_legacy_eval -p 5434:5432 -e POSTGRES_USER=mirix -e POSTGRES_PASSWORD=mirix -e POSTGRES_DB=mirix -v mirix_pgvector_legacy_eval_data:/var/lib/postgresql/data pgvector/pgvector:pg16` |
| New code | `evals/metaclaw/mirix_legacy_{client,evolver,manager}.py` |
| Adapter extension | `evals/metaclaw/format_adapter.py` (+`legacy_procedural_to_metaclaw`) |
| Driver patch | `evals/metaclaw/run_3day_eval.py` (+`--legacy` flag) |
| Tests | `evals/metaclaw/tests/test_mirix_legacy_*.py` (4 unit tests min) |
| Run output | `reports/<run-id>/{summary.json, summary.md, day_metrics.json, workspace/}` |
| Result archive | `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.{md,json}` |
| Three-way comparison | `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-three-way-comparison.md` |
| Spec (this doc) | `docs/superpowers/specs/2026-05-12-original-mirix-3day-design.md` |
| Plan | `docs/superpowers/plans/2026-05-12-original-mirix-3day.md` (next step) |

## 11. Success Criteria

1. `eval/original-mirix-3day` builds, server boots on 8532, postgres on 5434.
2. 3-day run completes with no uncaught exceptions across 30+ rounds.
3. `procedural_memory` table has ≥ 1 row per day under user `eval-legacy-3day` (R1 not silently firing).
4. Three-way comparison table is published and committed to `docs/superpowers/eval-results/`.
5. Result interpretation explicitly attributes any delta vs feat/skill-evolve to the schema/prompt/tools axis, not to confounders.
