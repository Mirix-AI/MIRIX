# Original-MIRIX 3-Day MetaClaw Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut a clean branch from `main`, port the existing MetaClaw eval harness onto it, plug a thin "legacy" adapter into the original MIRIX procedural-memory REST endpoints (`/v1/memory/add_sync` + `/v1/memory/search`), and run the same MetaClaw bench `day01..day03` as the feat/skill-evolve run — producing a publishable A/B/C comparison.

**Architecture:** New branch `eval/original-mirix-3day` from main; cherry-imported `evals/metaclaw/` directory from feat/skill-evolve; three new adapter files (`mirix_legacy_client.py`, `mirix_legacy_evolver.py`, `mirix_legacy_manager.py`) wire the existing `round_runner` agent loop to the legacy endpoints; driver gains a `--legacy` flag; an isolated postgres container on port 5434 + legacy server on port 8532 keep the run hermetic.

**Tech Stack:** Python 3.10+, httpx, pytest, FastAPI (MIRIX server), pgvector/pg16 (legacy db), Docker, OpenRouter (chat `openai/gpt-5.2`, embed `google/gemini-embedding-001`), MetaClaw bench.

---

## File Structure

**Created on `eval/original-mirix-3day`:**

- `evals/metaclaw/mirix_legacy_client.py` — async httpx wrapper around `/v1/memory/add_sync` + `/v1/memory/search`, with `meta_agent_id` bootstrap
- `evals/metaclaw/mirix_legacy_evolver.py` — `LegacyMirixEvolver(SkillEvolver)`, evolves by POSTing each round's serialised message
- `evals/metaclaw/mirix_legacy_manager.py` — `LegacyMirixManager(SkillManager)`, retrieves by GETting `/v1/memory/search`
- `evals/metaclaw/tests/test_format_adapter_legacy.py` — unit tests for the new format mapping
- `evals/metaclaw/tests/test_mirix_legacy_client.py` — unit tests for the new client
- `evals/metaclaw/tests/test_mirix_legacy_evolver.py` — unit tests for the new evolver
- `evals/metaclaw/tests/test_mirix_legacy_manager.py` — unit tests for the new manager

**Modified on `eval/original-mirix-3day`:**

- `evals/metaclaw/format_adapter.py` — append `legacy_procedural_to_metaclaw`
- `evals/metaclaw/run_3day_eval.py` — add `--legacy` flag and branch the MIRIX wiring

**Imported from feat/skill-evolve (read-only on this branch):**

- All of `evals/metaclaw/` scaffold (`round_runner.py`, `llm_config_helpers.py`, `format_adapter.py`, etc.)
- `docs/superpowers/specs/2026-05-12-original-mirix-3day-design.md`

**Result artefacts (created during execution, not in code):**

- `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.{md,json}`
- `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-three-way-comparison.md`

---

## Conventions

- **Commit prefix**: `[VEPEAGE-000] eval-legacy:` for all commits in this plan
- **Working directory**: assume `/Users/nanjiayan/Desktop/awesome_agent/MIRIX` (repo root)
- **MIRIX editable install**: `pip install -e .` must be in effect; the `mirix` package must resolve to this checkout
- **OpenRouter env**: `OPENAI_API_KEY` and `OPENAI_API_BASE` (set to `https://openrouter.ai/api/v1`) must be present
- **Postgres password**: `mirix` / `mirix` / db `mirix` (matches existing `docker/env.example`)

---

## Task 1: Cut branch + import scaffold

**Files:**
- New branch: `eval/original-mirix-3day` from `main` (HEAD `d6e7c14`)
- Imported (one commit): `evals/metaclaw/**`, `docs/superpowers/specs/2026-05-12-original-mirix-3day-design.md`

- [ ] **Step 1: Confirm we are on the repo root and there's no dirty state in tracked files**

Run:
```bash
cd /Users/nanjiayan/Desktop/awesome_agent/MIRIX
git status --short
```

Expected: untracked files only (CLAUDE.local.md, examples/, evals/test_*.py). Tracked files clean.

- [ ] **Step 2: Cut the new branch from main**

Run:
```bash
git fetch origin
git checkout main
git checkout -b eval/original-mirix-3day
```

Expected: branch switched to `eval/original-mirix-3day`. HEAD should be `d6e7c14` (`git log -1 --oneline`).

- [ ] **Step 3: Copy the eval scaffold from feat/skill-evolve via git checkout (preserves history line)**

Run:
```bash
git checkout feat/skill-evolve -- evals/metaclaw/ docs/superpowers/specs/2026-05-12-original-mirix-3day-design.md
```

Expected: `evals/metaclaw/` directory now staged with all `.py` files; spec staged.

- [ ] **Step 4: Also copy the gitignore entries for reports/ and third_party/**

Check that `.gitignore` already excludes `third_party/` and `evals/metaclaw/reports/`:

Run:
```bash
grep -E "third_party|metaclaw/reports" .gitignore || true
```

If neither line is present, append them:
```bash
printf "\n# Eval harness outputs\nthird_party/\nevals/metaclaw/reports/\n" >> .gitignore
git add .gitignore
```

- [ ] **Step 5: Stage and commit**

Run:
```bash
git add evals/metaclaw/ docs/superpowers/specs/2026-05-12-original-mirix-3day-design.md
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: import MetaClaw scaffold + spec from feat/skill-evolve"
```

Expected: a single commit on `eval/original-mirix-3day` containing the whole `evals/metaclaw/` tree + spec.

- [ ] **Step 6: Verify import**

Run:
```bash
ls evals/metaclaw/ | sort
```

Expected output (one per line, in some order):
```
__init__.py
format_adapter.py
llm_config_helpers.py
mirix_client.py
mirix_skill_evolver.py
mirix_skill_manager.py
round_runner.py
run_3day_eval.py
tests
```

`evals/metaclaw/tests/` should contain at least `__init__.py`, `test_format_adapter.py`, `test_mirix_client.py`, `test_mirix_skill_evolver.py`, `test_mirix_skill_manager.py`, `test_round_runner.py`.

- [ ] **Step 7: Smoke-import the scaffold under main's mirix code**

Run:
```bash
python -c "from evals.metaclaw import round_runner, format_adapter, llm_config_helpers; print('ok')"
```

Expected: `ok` (no ImportError). If `mirix` is not editable-installed in the active venv, `pip install -e .` first.

- [ ] **Step 8: Note — existing skill-based tests are EXPECTED to fail against main**

The scaffold includes `test_mirix_skill_evolver.py`, `test_mirix_skill_manager.py`, `test_mirix_client.py`. Those tests use httpx.MockTransport, so they do NOT need a live `/v1/skills` endpoint. They should still pass purely on shape.

Run:
```bash
pytest -q evals/metaclaw/tests/test_format_adapter.py evals/metaclaw/tests/test_round_runner.py evals/metaclaw/tests/test_mirix_client.py evals/metaclaw/tests/test_mirix_skill_evolver.py evals/metaclaw/tests/test_mirix_skill_manager.py
```

Expected: all green. If any fail because of MIRIX api drift on main, note them in the commit message of step 5 and proceed — we only depend on the bench-agnostic files (`round_runner.py`, `format_adapter.py`, `llm_config_helpers.py`).

---

## Task 2: Extend `format_adapter.py` with `legacy_procedural_to_metaclaw`

**Files:**
- Modify: `evals/metaclaw/format_adapter.py` (append one function)
- Test: `evals/metaclaw/tests/test_format_adapter_legacy.py` (new)

- [ ] **Step 1: Write the failing test**

Create `evals/metaclaw/tests/test_format_adapter_legacy.py`:
```python
"""Unit tests for legacy_procedural_to_metaclaw — the adapter that maps
main-branch procedural_memory rows ({summary, steps, entry_type}) to the
metaclaw skill-shape ({name, description, content, category}).
"""
from evals.metaclaw.format_adapter import legacy_procedural_to_metaclaw


def test_full_row_maps_all_fields():
    row = {
        "id": "proc-1",
        "summary": "Format dates as ISO 8601",
        "steps": "1. Identify date 2. Convert to YYYY-MM-DDTHH:MM:SSZ",
        "entry_type": "guide",
    }
    out = legacy_procedural_to_metaclaw(row)
    assert out == {
        "name": "guide",
        "description": "Format dates as ISO 8601",
        "content": "1. Identify date 2. Convert to YYYY-MM-DDTHH:MM:SSZ",
        "category": "guide",
    }


def test_missing_entry_type_defaults_to_procedure():
    out = legacy_procedural_to_metaclaw({"summary": "x", "steps": "y"})
    assert out["name"] == "procedure"
    assert out["category"] == "procedure"


def test_missing_summary_and_steps_yield_empty_strings():
    out = legacy_procedural_to_metaclaw({"entry_type": "workflow"})
    assert out == {
        "name": "workflow",
        "description": "",
        "content": "",
        "category": "workflow",
    }


def test_null_values_treated_as_missing():
    out = legacy_procedural_to_metaclaw(
        {"summary": None, "steps": None, "entry_type": None}
    )
    assert out == {
        "name": "procedure",
        "description": "",
        "content": "",
        "category": "procedure",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest -q evals/metaclaw/tests/test_format_adapter_legacy.py
```

Expected: ImportError (`legacy_procedural_to_metaclaw` not in `format_adapter`).

- [ ] **Step 3: Implement**

Append to `evals/metaclaw/format_adapter.py`:
```python


def legacy_procedural_to_metaclaw(row: dict) -> dict:
    """Map a main-branch procedural_memory row to the metaclaw skill-shape.

    Old schema fields: summary, steps, entry_type.
    Target shape (same as mirix_to_metaclaw): name, description, content, category.
    """
    entry_type = row.get("entry_type") or "procedure"
    return {
        "name": entry_type,
        "description": row.get("summary") or "",
        "content": row.get("steps") or "",
        "category": entry_type,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest -q evals/metaclaw/tests/test_format_adapter_legacy.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/metaclaw/format_adapter.py evals/metaclaw/tests/test_format_adapter_legacy.py
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: add legacy_procedural_to_metaclaw adapter"
```

---

## Task 3: `mirix_legacy_client.py`

**Files:**
- Create: `evals/metaclaw/mirix_legacy_client.py`
- Test: `evals/metaclaw/tests/test_mirix_legacy_client.py`

- [ ] **Step 1: Write the failing test**

Create `evals/metaclaw/tests/test_mirix_legacy_client.py`:
```python
"""Unit tests for LegacyMirixClient using httpx.MockTransport."""
import json

import httpx
import pytest

from evals.metaclaw.mirix_legacy_client import (
    DEFAULT_CLIENT_ID,
    DEFAULT_META_AGENT_NAME,
    LegacyMirixClient,
)


def _mock_transport(handler):
    return httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
        headers={"X-Client-Id": DEFAULT_CLIENT_ID},
    )


@pytest.mark.asyncio
async def test_resolve_meta_agent_id_picks_correct_agent():
    """GET /v1/agents must be filtered to the meta_memory_agent row."""
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["path"] = req.url.path
        return httpx.Response(
            200,
            json=[
                {"id": "agent-1", "name": "core_memory_agent"},
                {"id": "agent-2", "name": DEFAULT_META_AGENT_NAME},
                {"id": "agent-3", "name": "episodic_memory_agent"},
            ],
        )

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        agent_id = await cli._resolve_meta_agent_id()
        assert agent_id == "agent-2"
        assert captured == {"method": "GET", "path": "/v1/agents"}


@pytest.mark.asyncio
async def test_resolve_meta_agent_raises_when_missing():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "agent-1", "name": "core_memory_agent"}])

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        with pytest.raises(RuntimeError, match="meta_memory_agent"):
            await cli._resolve_meta_agent_id()


@pytest.mark.asyncio
async def test_add_memory_posts_to_add_sync_with_correct_body():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/agents":
            return httpx.Response(200, json=[{"id": "ma-1", "name": DEFAULT_META_AGENT_NAME}])
        captured["path"] = req.url.path
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"success": True, "status": "processed"})

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client, user_id="eval-legacy-3day")
        out = await cli.add_memory("round-1 transcript text")

    assert captured["path"] == "/v1/memory/add_sync"
    assert captured["body"] == {
        "meta_agent_id": "ma-1",
        "messages": [{"role": "user", "content": "round-1 transcript text"}],
        "user_id": "eval-legacy-3day",
        "chaining": True,
    }
    assert out["status"] == "processed"


@pytest.mark.asyncio
async def test_search_procedural_sends_correct_query_params():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["params"] = dict(req.url.params)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "memory_type": "procedural",
                        "content": {
                            "id": "p-1",
                            "summary": "Format dates as ISO 8601",
                            "steps": "Use YYYY-MM-DD",
                            "entry_type": "guide",
                        },
                    }
                ]
            },
        )

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client, user_id="eval-legacy-3day")
        rows = await cli.search_procedural(query="what is today's date format", limit=6)

    assert captured["path"] == "/v1/memory/search"
    assert captured["params"]["memory_type"] == "procedural"
    assert captured["params"]["search_method"] == "bm25"
    assert captured["params"]["search_field"] == "summary"
    assert captured["params"]["query"] == "what is today's date format"
    assert captured["params"]["limit"] == "6"
    assert captured["params"]["user_id"] == "eval-legacy-3day"
    assert rows == [
        {
            "id": "p-1",
            "summary": "Format dates as ISO 8601",
            "steps": "Use YYYY-MM-DD",
            "entry_type": "guide",
        }
    ]


@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        assert await cli.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_on_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with _mock_transport(handler) as client:
        cli = LegacyMirixClient(_client=client)
        assert await cli.health() is False
```

Note: the test for `search_procedural` assumes the server returns `{"results": [{"memory_type": "procedural", "content": {…procedural row…}}]}`. This matches main's `rest_api.py:3234-3263`. The client unwraps the `content` dicts.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest -q evals/metaclaw/tests/test_mirix_legacy_client.py
```

Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement**

Create `evals/metaclaw/mirix_legacy_client.py`:
```python
"""Async wrapper around the *original* MIRIX REST endpoints used by this
eval harness — the pre-skill-evolve world.

Endpoints used:
    GET  /v1/agents                        — find meta_memory_agent id
    POST /v1/memory/add_sync               — synchronously feed a round to
                                              meta_agent (routes to
                                              procedural_memory_agent which
                                              writes summary+steps rows)
    GET  /v1/memory/search?memory_type=procedural
                                           — BM25 retrieval over procedural
                                              memory by summary
    GET  /health                            — liveness probe

We deliberately use /v1/memory/add_sync (not /v1/memory/add) — the latter
queues via Kafka and would let writes lag a round behind retrieves,
silently degrading this arm to the no-skills baseline.
"""
from __future__ import annotations

from typing import Any

import httpx

DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_META_AGENT_NAME = "meta_memory_agent"


class LegacyMirixClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8532",
        user_id: str = "eval-legacy-3day",
        client_id: str = DEFAULT_CLIENT_ID,
        timeout: float = 600.0,
        _client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.client_id = client_id
        self._timeout = timeout
        self._client = _client
        self._owns_client = _client is None
        self._meta_agent_id: str | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers={"X-Client-Id": self.client_id},
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _resolve_meta_agent_id(self) -> str:
        """One-shot bootstrap: list agents, pick the one named meta_memory_agent.
        Cached on first success."""
        if self._meta_agent_id is not None:
            return self._meta_agent_id
        client = await self._get_client()
        resp = await client.get("/v1/agents")
        resp.raise_for_status()
        agents = resp.json()
        for a in agents:
            if a.get("name") == DEFAULT_META_AGENT_NAME:
                self._meta_agent_id = a["id"]
                return self._meta_agent_id
        raise RuntimeError(
            f"No agent named {DEFAULT_META_AGENT_NAME!r} found on the server. "
            f"Call POST /v1/agents/meta/initialize first."
        )

    async def add_memory(self, message_text: str) -> dict[str, Any]:
        """POST /v1/memory/add_sync. Synchronous — by the time this returns,
        any procedural-memory writes from the meta_agent are durable."""
        agent_id = await self._resolve_meta_agent_id()
        client = await self._get_client()
        resp = await client.post(
            "/v1/memory/add_sync",
            json={
                "meta_agent_id": agent_id,
                "messages": [{"role": "user", "content": message_text}],
                "user_id": self.user_id,
                "chaining": True,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def search_procedural(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """GET /v1/memory/search?memory_type=procedural&search_method=bm25
        &search_field=summary&query=...&limit=...&user_id=..."""
        client = await self._get_client()
        resp = await client.get(
            "/v1/memory/search",
            params={
                "memory_type": "procedural",
                "search_method": "bm25",
                "search_field": "summary",
                "query": query,
                "limit": limit,
                "user_id": self.user_id,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        results = body.get("results", body.get("memories", []))
        rows: list[dict[str, Any]] = []
        for r in results:
            if r.get("memory_type") and r.get("memory_type") != "procedural":
                continue
            content = r.get("content", r)
            rows.append(content)
        return rows

    async def health(self) -> bool:
        client = await self._get_client()
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest -q evals/metaclaw/tests/test_mirix_legacy_client.py
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/metaclaw/mirix_legacy_client.py evals/metaclaw/tests/test_mirix_legacy_client.py
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: add LegacyMirixClient (add_sync + memory/search wrappers)"
```

---

## Task 4: `mirix_legacy_evolver.py`

**Files:**
- Create: `evals/metaclaw/mirix_legacy_evolver.py`
- Test: `evals/metaclaw/tests/test_mirix_legacy_evolver.py`

- [ ] **Step 1: Write the failing test**

Create `evals/metaclaw/tests/test_mirix_legacy_evolver.py`:
```python
"""Unit tests for LegacyMirixEvolver."""
import pytest

from evals.metaclaw.format_adapter import RoundResult
from evals.metaclaw.mirix_legacy_evolver import LegacyMirixEvolver


class FakeMirix:
    def __init__(self):
        self.calls: list[str] = []

    async def add_memory(self, message_text: str) -> dict:
        self.calls.append(message_text)
        return {"status": "processed"}


def _round(rid: str = "r1") -> RoundResult:
    return RoundResult(
        round_id=rid,
        round_type="multi_choice",
        question=f"Question for {rid}?",
        final_answer="A",
        reward=1,
        eval_outcome="correct",
        feedback="",
        transcript="(transcript)",
        error=None,
    )


@pytest.mark.asyncio
async def test_should_evolve_always_true():
    evo = LegacyMirixEvolver(mirix=FakeMirix())
    assert evo.should_evolve() is True
    assert evo.should_evolve(some_kwarg=123) is True


@pytest.mark.asyncio
async def test_evolve_async_posts_one_message_per_round():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=mirix)
    out = await evo.evolve_async([_round("r1"), _round("r2"), _round("r3")])
    assert len(mirix.calls) == 3
    assert "r1" in mirix.calls[0]
    assert "r3" in mirix.calls[2]
    assert out == {"sent": 3}


@pytest.mark.asyncio
async def test_evolve_async_empty_list_is_noop():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=mirix)
    out = await evo.evolve_async([])
    assert mirix.calls == []
    assert out == {"sent": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest -q evals/metaclaw/tests/test_mirix_legacy_evolver.py
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `evals/metaclaw/mirix_legacy_evolver.py`:
```python
"""LegacyMirixEvolver — feeds each round's serialised message to the
original-mirix `meta_agent` via POST /v1/memory/add_sync. Subclasses
metaclaw.skill_evolver.SkillEvolver for interface parity but bypasses
parent __init__ since we don't need any of its state.
"""
from __future__ import annotations

from typing import Any

from metaclaw.skill_evolver import SkillEvolver

from evals.metaclaw.format_adapter import RoundResult, round_to_message
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient


class LegacyMirixEvolver(SkillEvolver):
    def __init__(self, mirix: LegacyMirixClient):
        # Skip parent __init__ — we don't need its file-system / LLM state.
        self.mirix = mirix

    def should_evolve(self, *args: Any, **kwargs: Any) -> bool:
        return True

    async def evolve_async(self, rounds: list[RoundResult]) -> dict[str, Any]:
        for r in rounds:
            await self.mirix.add_memory(round_to_message(r))
        return {"sent": len(rounds)}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest -q evals/metaclaw/tests/test_mirix_legacy_evolver.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/metaclaw/mirix_legacy_evolver.py evals/metaclaw/tests/test_mirix_legacy_evolver.py
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: add LegacyMirixEvolver"
```

---

## Task 5: `mirix_legacy_manager.py`

**Files:**
- Create: `evals/metaclaw/mirix_legacy_manager.py`
- Test: `evals/metaclaw/tests/test_mirix_legacy_manager.py`

- [ ] **Step 1: Write the failing test**

Create `evals/metaclaw/tests/test_mirix_legacy_manager.py`:
```python
"""Unit tests for LegacyMirixManager."""
import pytest

from evals.metaclaw.mirix_legacy_manager import (
    DEFAULT_TOP_K,
    LegacyMirixManager,
)


class FakeMirix:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, int]] = []

    async def search_procedural(self, query: str, limit: int = 6):
        self.calls.append((query, limit))
        return self.rows


@pytest.mark.asyncio
async def test_retrieve_async_maps_rows_to_metaclaw_shape():
    rows = [
        {
            "summary": "Format dates as ISO 8601",
            "steps": "Use YYYY-MM-DDTHH:MM:SSZ",
            "entry_type": "guide",
        },
        {
            "summary": "Save under workspace/answers/",
            "steps": "Append answer.txt",
            "entry_type": "workflow",
        },
    ]
    mgr = LegacyMirixManager(mirix=FakeMirix(rows))
    out = await mgr.retrieve_async("how should I format dates?")
    assert out == [
        {
            "name": "guide",
            "description": "Format dates as ISO 8601",
            "content": "Use YYYY-MM-DDTHH:MM:SSZ",
            "category": "guide",
        },
        {
            "name": "workflow",
            "description": "Save under workspace/answers/",
            "content": "Append answer.txt",
            "category": "workflow",
        },
    ]


@pytest.mark.asyncio
async def test_retrieve_async_passes_query_and_default_top_k():
    fake = FakeMirix([])
    mgr = LegacyMirixManager(mirix=fake)
    await mgr.retrieve_async("Q")
    assert fake.calls == [("Q", DEFAULT_TOP_K)]


@pytest.mark.asyncio
async def test_retrieve_async_passes_custom_top_k():
    fake = FakeMirix([])
    mgr = LegacyMirixManager(mirix=fake)
    await mgr.retrieve_async("Q", top_k=3)
    assert fake.calls == [("Q", 3)]


@pytest.mark.asyncio
async def test_retrieve_async_empty_rows_yields_empty_list():
    mgr = LegacyMirixManager(mirix=FakeMirix([]))
    out = await mgr.retrieve_async("anything")
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest -q evals/metaclaw/tests/test_mirix_legacy_manager.py
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `evals/metaclaw/mirix_legacy_manager.py`:
```python
"""LegacyMirixManager — retrieves procedural memories from main-branch
MIRIX via GET /v1/memory/search and maps them into the metaclaw skill-shape
that round_runner expects.
"""
from __future__ import annotations

from typing import Any

from metaclaw.skill_manager import SkillManager

from evals.metaclaw.format_adapter import legacy_procedural_to_metaclaw
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient

DEFAULT_TOP_K = 6


class LegacyMirixManager(SkillManager):
    def __init__(self, mirix: LegacyMirixClient):
        # Skip parent __init__ — we don't need its file-system state.
        self.mirix = mirix

    async def retrieve_async(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        rows = await self.mirix.search_procedural(query=query, limit=top_k)
        return [legacy_procedural_to_metaclaw(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest -q evals/metaclaw/tests/test_mirix_legacy_manager.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/metaclaw/mirix_legacy_manager.py evals/metaclaw/tests/test_mirix_legacy_manager.py
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: add LegacyMirixManager"
```

---

## Task 6: Patch `run_3day_eval.py` with `--legacy` flag

**Files:**
- Modify: `evals/metaclaw/run_3day_eval.py`

- [ ] **Step 1: Read the existing driver to find the integration points**

Run:
```bash
grep -nE "(--no-skills|use_mirix|MirixClient\(|MirixSkillEvolver\(|MirixSkillManager\(|mirix_url|argparse|add_argument)" evals/metaclaw/run_3day_eval.py
```

Take note of:
- The `argparse` section where `--no-skills` is defined → add `--legacy` nearby
- The `use_mirix` branch where `MirixClient` / `MirixSkillEvolver` / `MirixSkillManager` are constructed → add the legacy alternative
- The default value for `--mirix-url` (currently `http://127.0.0.1:8531`)

- [ ] **Step 2: Add `--legacy` flag in argparse, mutually exclusive with `--no-skills`**

Find the argparse block. Add this argument near `--no-skills`:
```python
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use original (pre-skill-evolve) MIRIX procedural memory endpoints "
             "(/v1/memory/add_sync + /v1/memory/search) instead of /v1/skills/*. "
             "Mutually exclusive with --no-skills.",
    )
```

Immediately after `args = parser.parse_args()`, enforce mutual exclusion and adjust the default mirix_url:
```python
    if args.legacy and args.no_skills:
        parser.error("--legacy and --no-skills are mutually exclusive")
    # Legacy server runs on a different port to avoid colliding with the new one
    if args.legacy and args.mirix_url == "http://127.0.0.1:8531":
        args.mirix_url = "http://127.0.0.1:8532"
```

(If the existing `--mirix-url` default uses a different string, substitute it.)

- [ ] **Step 3: Add imports at the top of the file (near the existing MIRIX adapter imports)**

```python
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient
from evals.metaclaw.mirix_legacy_evolver import LegacyMirixEvolver
from evals.metaclaw.mirix_legacy_manager import LegacyMirixManager
```

- [ ] **Step 4: Branch the MIRIX wiring**

Locate the block that looks roughly like:
```python
use_mirix = not args.dry_run and not args.no_skills
if use_mirix:
    mirix = MirixClient(base_url=args.mirix_url, user_id=args.user_id, timeout=args.mirix_timeout)
    evolver = MirixSkillEvolver(mirix=mirix)
    skill_mgr = MirixSkillManager(mirix=mirix)
```

Restructure to:
```python
use_mirix = not args.dry_run and not args.no_skills
mirix = None
evolver = None
skill_mgr = None
if use_mirix:
    if args.legacy:
        mirix = LegacyMirixClient(
            base_url=args.mirix_url,
            user_id=args.user_id,
            timeout=args.mirix_timeout,
        )
        evolver = LegacyMirixEvolver(mirix=mirix)
        skill_mgr = LegacyMirixManager(mirix=mirix)
    else:
        mirix = MirixClient(
            base_url=args.mirix_url,
            user_id=args.user_id,
            timeout=args.mirix_timeout,
        )
        evolver = MirixSkillEvolver(mirix=mirix)
        skill_mgr = MirixSkillManager(mirix=mirix)
```

- [ ] **Step 5: Ensure the run-id in `summary.{json,md}` records the arm**

In the run-id / summary scaffolding (search for `run_id` or `arm` in the driver), add an `arm` field set to:
- `"legacy"` if `args.legacy`
- `"no_skills"` if `args.no_skills`
- `"skills"` otherwise

If the driver doesn't already record `arm`, add it to the summary dict written to `summary.json` and the markdown body of `summary.md`. This is essential for downstream comparison.

Concrete patch — search for the `summary = {` (or equivalent dict construction) and add:
```python
        "arm": "legacy" if args.legacy else ("no_skills" if args.no_skills else "skills"),
```

If `summary.md` is written, prepend a line like `**arm**: <arm>` after the title.

- [ ] **Step 6: Sanity check — dry-run the driver under `--legacy`**

Run:
```bash
python -m evals.metaclaw.run_3day_eval --legacy --dry-run --days day01 --max-rounds 1
```

Expected: completes without exception; reports `dry-run` summary; the run-id directory under `evals/metaclaw/reports/` records `"arm": "legacy"`. The driver should NOT try to contact MIRIX in `--dry-run` (it never does in skill mode either — confirm by checking the code).

- [ ] **Step 7: Commit**

```bash
git add evals/metaclaw/run_3day_eval.py
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: add --legacy flag to run_3day_eval driver"
```

---

## Task 7: Bring up isolated legacy infrastructure

**Files:**
- (No code) — runbook task; produces a running postgres + MIRIX server on host

- [ ] **Step 1: Confirm prerequisites**

Run:
```bash
docker --version
python --version           # 3.10+
echo "${OPENAI_API_KEY:0:7}..."   # must be non-empty
echo "${OPENAI_API_BASE}"          # must be https://openrouter.ai/api/v1
```

If `OPENAI_API_BASE` is unset:
```bash
export OPENAI_API_BASE=https://openrouter.ai/api/v1
```

- [ ] **Step 2: Stop any existing eval postgres/server to free ports**

Run:
```bash
docker ps --filter "name=mirix_pgvector" --format "{{.Names}} {{.Ports}}"
lsof -nP -iTCP:8531 -sTCP:LISTEN || true
lsof -nP -iTCP:8532 -sTCP:LISTEN || true
```

If any feat/skill-evolve MIRIX server is still listening on 8531, you may leave it — we'll use 8532.

- [ ] **Step 3: Spin up isolated postgres on port 5434**

Run:
```bash
docker volume create mirix_pgvector_legacy_eval_data
docker run -d \
  --name mirix_pgvector_legacy_eval \
  -p 5434:5432 \
  -e POSTGRES_USER=mirix \
  -e POSTGRES_PASSWORD=mirix \
  -e POSTGRES_DB=mirix \
  -v mirix_pgvector_legacy_eval_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

Wait ~10s, then verify:
```bash
docker exec mirix_pgvector_legacy_eval pg_isready -U mirix -d mirix
```

Expected: `accepting connections`.

- [ ] **Step 4: Start legacy MIRIX server**

In a separate terminal, set the MIRIX db URI to the legacy postgres and start the server on port 8532:
```bash
export MIRIX_PG_URI="postgresql+asyncpg://mirix:mirix@127.0.0.1:5434/mirix"
export OPENAI_API_BASE=https://openrouter.ai/api/v1
# OPENAI_API_KEY must already be in env
python scripts/start_server.py --port 8532
```

Expected: server logs end with `Uvicorn running on http://0.0.0.0:8532`. On first boot, DDL is created against the legacy postgres.

- [ ] **Step 5: Bootstrap default client + meta_agent**

From the main terminal:
```bash
curl -s -X POST http://127.0.0.1:8532/v1/clients/create_or_get \
  -H 'Content-Type: application/json' \
  -d '{"id": "client-00000000-0000-4000-8000-000000000000", "name": "eval-default"}'
echo
curl -s -X POST http://127.0.0.1:8532/v1/agents/meta/initialize \
  -H 'Content-Type: application/json' \
  -H "X-Client-Id: client-00000000-0000-4000-8000-000000000000" \
  -d '{}'
echo
curl -s http://127.0.0.1:8532/v1/agents \
  -H "X-Client-Id: client-00000000-0000-4000-8000-000000000000" | python -m json.tool | head -30
```

Expected: the last command's output includes an entry with `"name": "meta_memory_agent"` (and 6+ child agents). If meta_memory_agent is missing, re-issue the `/v1/agents/meta/initialize` POST and re-list.

- [ ] **Step 6: Smoke the legacy client end-to-end against the live server**

Run an inline Python smoke (no commit):
```bash
python - <<'PY'
import asyncio
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient

async def main():
    cli = LegacyMirixClient(base_url="http://127.0.0.1:8532", user_id="smoke-legacy")
    assert await cli.health() is True, "health failed"
    aid = await cli._resolve_meta_agent_id()
    print("meta_agent_id:", aid)
    out = await cli.add_memory("Smoke test: please remember that ISO 8601 dates look like YYYY-MM-DDTHH:MM:SSZ.")
    print("add_memory:", out.get("status"))
    rows = await cli.search_procedural(query="ISO 8601 date format", limit=3)
    print("rows returned:", len(rows))
    for r in rows[:3]:
        print(" -", r.get("summary"))
    await cli.aclose()

asyncio.run(main())
PY
```

Expected:
- `meta_agent_id: <some agent id>`
- `add_memory: processed`
- `rows returned: ≥ 0` (may be 0 the first time if the agent decided not to store; will be ≥ 1 after a few rounds)

- [ ] **Step 7: Document the bring-up steps (no commit yet)**

Keep a scratch note: command lines, postgres container name, server port, meta_agent_id. These go into the result-write-up commit later.

---

## Task 8: Single-round live smoke + R1 tripwire

**Files:**
- (no code unless R1 fires)

- [ ] **Step 1: Run one live round on day01 with `--legacy`**

```bash
python -m evals.metaclaw.run_3day_eval \
  --legacy \
  --days day01 \
  --max-rounds 1 \
  --top-k 6 \
  --user-id eval-legacy-3day \
  --mirix-url http://127.0.0.1:8532
```

Expected: the driver completes 1 round; reports under `evals/metaclaw/reports/<run-id>/` exist; `summary.json` records `"arm": "legacy"`.

- [ ] **Step 2: Inspect db for a row under the legacy user**

```bash
docker exec -e PGPASSWORD=mirix mirix_pgvector_legacy_eval \
  psql -U mirix -d mirix -c \
  "SELECT id, entry_type, left(summary,80) AS summary, left(steps,80) AS steps FROM procedural_memory WHERE user_id = 'eval-legacy-3day' ORDER BY created_at DESC LIMIT 5;"
```

Expected: ≥ 1 row.

- [ ] **Step 3: If 0 rows — R1 tripwire fires**

Apply D10 fallback: prepend a reflection prefix inside `mirix_legacy_evolver.py`:

```python
REFLECTION_PREFIX = (
    "Reflect on the round below and extract any reusable procedural knowledge "
    "into procedural memory.\n\n"
)

class LegacyMirixEvolver(SkillEvolver):
    ...
    async def evolve_async(self, rounds: list[RoundResult]) -> dict[str, Any]:
        for r in rounds:
            await self.mirix.add_memory(REFLECTION_PREFIX + round_to_message(r))
        return {"sent": len(rounds)}
```

Also update the corresponding test (`test_mirix_legacy_evolver.py`):
```python
async def test_evolve_async_prepends_reflection_prefix():
    mirix = FakeMirix()
    evo = LegacyMirixEvolver(mirix=FakeMirix())
    await evo.evolve_async([_round("r1")])
    assert mirix.calls[0].startswith(REFLECTION_PREFIX) or "Reflect on the round below" in mirix.calls[0]
```

Re-run the failing test, then the live smoke, then re-check the db. If still 0, escalate (see Risks R1 in the spec).

- [ ] **Step 4: If the tripwire fired, commit the fallback**

```bash
git add evals/metaclaw/mirix_legacy_evolver.py evals/metaclaw/tests/test_mirix_legacy_evolver.py
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: enable D10 reflection prefix (R1 tripwire fired)"
```

If the smoke succeeded without R1 firing, skip this commit.

---

## Task 9: Full 3-day run

**Files:**
- Output: `evals/metaclaw/reports/<run-id>/summary.{json,md}`, `day_metrics.json`, `workspace/`

- [ ] **Step 1: Launch the full run in the background**

```bash
python -m evals.metaclaw.run_3day_eval \
  --legacy \
  --days day01,day02,day03 \
  --max-rounds 12 \
  --max-turns 24 \
  --wallclock-cap 180 \
  --top-k 6 \
  --user-id eval-legacy-3day \
  --mirix-url http://127.0.0.1:8532 \
  --mirix-timeout 900 \
  --log-level INFO \
  2>&1 | tee /tmp/legacy_3day.log
```

Expected: runs ~25-45 minutes. During the run, you can `tail -f /tmp/legacy_3day.log`. Look for:
- `[day01] round 0/12 starting`
- `[day01] round X reward=1` or `reward=0`
- end-of-day summary lines
- `[day02] ...`, `[day03] ...`

- [ ] **Step 2: Verify completion**

When the run ends:
```bash
ls -t evals/metaclaw/reports/ | head -1
RUN_ID=$(ls -t evals/metaclaw/reports/ | head -1)
echo "Latest run: $RUN_ID"
cat evals/metaclaw/reports/$RUN_ID/summary.md
```

Expected: `summary.md` lists per-day mean reward and an overall mean.

- [ ] **Step 3: Sanity check the db got per-day writes**

```bash
docker exec -e PGPASSWORD=mirix mirix_pgvector_legacy_eval \
  psql -U mirix -d mirix -c \
  "SELECT date_trunc('day', created_at) AS day, count(*) FROM procedural_memory WHERE user_id = 'eval-legacy-3day' GROUP BY 1 ORDER BY 1;"
```

Expected: 1 or more rows for each of the run's 3 calendar days, OR (if R1 fired despite the prefix) at least one row total.

- [ ] **Step 4: Commit nothing yet — Task 10 packages the results**

---

## Task 10: Archive results under `docs/superpowers/eval-results/`

**Files:**
- Create: `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.md`
- Create: `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.json`

- [ ] **Step 1: Locate the latest run output**

```bash
RUN_ID=$(ls -t evals/metaclaw/reports/ | head -1)
echo $RUN_ID
test -f evals/metaclaw/reports/$RUN_ID/summary.json
```

- [ ] **Step 2: Copy `summary.json` as the canonical archived JSON**

```bash
cp evals/metaclaw/reports/$RUN_ID/summary.json \
   docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.json
```

- [ ] **Step 3: Author the human-readable summary markdown**

Create `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.md` with this template (fill the numbers from `summary.json` after copying):

```markdown
# MetaClaw 3-day eval — Original-MIRIX (pre-skill-evolve) arm

**Run id**: `<paste RUN_ID>`
**Date**: 2026-05-12
**Branch**: `eval/original-mirix-3day` (from main HEAD `d6e7c14`)
**Arm**: `legacy` (original MIRIX procedural memory: `summary+steps` schema, old `procedural_memory_insert/update` tools)
**Model**: OpenRouter `openai/gpt-5.2` (chat) + `google/gemini-embedding-001` (embedding)
**Bench**: `metaclaw-bench/eval/day01..day03` (P1 ISO 8601 preference arc)
**Grain**: 1 evolve message per round + 1 retrieve per round
**Postgres**: `mirix_pgvector_legacy_eval` (port 5434)
**MIRIX server**: port 8532
**Reflection prefix (D10)**: `<yes / no>`

## Per-day accuracy

| Day | Rounds | Correct | Accuracy |
|---|---|---|---|
| day01 | <X> | <Y> | <Z> |
| day02 | <X> | <Y> | <Z> |
| day03 | <X> | <Y> | <Z> |
| **mean** | | | **<MEAN>** |

## Procedural memory growth

| Day | New rows | Total rows (end of day) |
|---|---|---|
| day01 | <N1> | <N1> |
| day02 | <N2> | <N1+N2> |
| day03 | <N3> | <N1+N2+N3> |

(Pull these from `SELECT date_trunc('day', created_at) AS day, count(*) FROM procedural_memory WHERE user_id='eval-legacy-3day' GROUP BY 1`.)

## Notes

- (any observations — e.g., whether the old agent reliably called procedural_memory_insert, whether retrieval returned anything for early rounds, etc.)
```

- [ ] **Step 4: Stage and commit**

The `docs/superpowers/` tree is typically gitignored; force-add.
```bash
git add -f \
  docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.md \
  docs/superpowers/eval-results/2026-05-12-metaclaw-3day-legacy-original-mirix-summary.json
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: archive 3-day Original-MIRIX results"
```

---

## Task 11: Three-way comparison doc

**Files:**
- Create: `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-three-way-comparison.md`

- [ ] **Step 1: Gather the three sets of numbers**

Reference these existing files (already committed on `feat/skill-evolve`):
- feat/skill-evolve arm: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.{md,json}`
- no-skills baseline: `docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.{md,json}`
- legacy arm: just produced in Task 10

(If the feat/skill-evolve docs aren't yet on `eval/original-mirix-3day`, cherry-pick or copy them — they're needed for the comparison table to be in-tree.)

```bash
mkdir -p docs/superpowers/eval-results
git checkout feat/skill-evolve -- \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.md \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.json \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.md \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.json \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-comparison.md
```

- [ ] **Step 2: Author the three-way comparison**

Create `docs/superpowers/eval-results/2026-05-12-metaclaw-3day-three-way-comparison.md`:

```markdown
# MetaClaw 3-day eval — Three-way comparison

**Date**: 2026-05-12
**Bench**: `metaclaw-bench/eval/day01..day03` (P1 ISO 8601 preference arc)
**Model**: OpenRouter `openai/gpt-5.2` + `google/gemini-embedding-001`
**Agent loop / scorer**: identical across all three arms (`evals/metaclaw/round_runner.py`)

| Arm | Branch | Procedural memory | Evolve endpoint | Retrieve endpoint |
|---|---|---|---|---|
| **A. feat/skill-evolve** | `feat/skill-evolve` | `name/description/instructions/triggers/examples/version` schema, 5 CLI tools | `POST /v1/skills/evolve` | `GET /v1/skills?query=...` |
| **B. original MIRIX** | `eval/original-mirix-3day` (from `main`) | `summary/steps/entry_type` schema, old `procedural_memory_insert/update` tools | `POST /v1/memory/add_sync` | `GET /v1/memory/search?memory_type=procedural` |
| **C. no skills** | `feat/skill-evolve --no-skills` | (none — cold every round) | (none) | (none) |

## Per-day accuracy

| Day | Rounds | A. feat/skill-evolve | B. original-MIRIX | C. no-skills | Δ (A − C) | Δ (B − C) | Δ (A − B) |
|---|---|---|---|---|---|---|---|
| day01 | … | … | … | … | … | … | … |
| day02 | … | … | … | … | … | … | … |
| day03 | … | … | … | … | … | … | … |
| **mean** | | **…** | **…** | **…** | **…** | **…** | **…** |

## Interpretation

- (A vs C) — net lift of the **new skill-evolve stack** over no procedural memory.
- (B vs C) — net lift of the **original MIRIX** procedural memory over no procedural memory.
- (A vs B) — clean attribution of the **skill-evolve refactor** (schema + agent prompt + tools) over the original procedural memory stack, with every other variable held constant.

Any non-zero Δ(A − B) on day02+ is the quantity this evaluation set out to measure.

## Caveats

- Single seed. CIs not estimated (follow-up).
- Day01 is cold-start for A and B; numbers there are sensitive to LLM noise more than to memory differences.
- If the D10 reflection prefix was applied in arm B, note it here — it would mean arm B got a small prompt-level boost that arm A did not.
```

- [ ] **Step 3: Fill in the numbers and write the interpretation paragraph**

Pull each row's accuracy from the three `summary.json` files and write 4-6 sentences of interpretation under `## Interpretation` keyed to the actual deltas observed.

- [ ] **Step 4: Stage and commit**

```bash
git add -f \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.md \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.json \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.md \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-baseline-no-mirix-summary.json \
  docs/superpowers/eval-results/2026-05-08-metaclaw-3day-comparison.md \
  docs/superpowers/eval-results/2026-05-12-metaclaw-3day-three-way-comparison.md
git -c commit.gpgsign=false commit -m "[VEPEAGE-000] eval-legacy: three-way A/B/C comparison vs feat/skill-evolve and no-skills baseline"
```

---

## Completion Criteria

- `eval/original-mirix-3day` branch contains Tasks 1-6 commits + Task 10 + Task 11 commits.
- All unit tests for the 3 new adapter files (`test_mirix_legacy_*.py`) pass green.
- `evals/metaclaw/reports/<run-id>/summary.json` records `"arm": "legacy"`.
- `procedural_memory` table on `mirix_pgvector_legacy_eval` contains ≥ 1 row per run-day under user `eval-legacy-3day` (R1 not silently firing).
- Three-way comparison doc is published and committed.

## Self-Review Notes

**Spec coverage:** All 10 locked decisions in the spec are covered: D1 grain (T6+T9 use one evolve per round), D2 branch+db (T1+T7), D3 reuse (T1 cherry-pick), D4 three files (T3+T4+T5), D5 same shape (T2), D6 ports (T6+T7), D7 model (T7+T9), D8 cold-start (T9 fresh db), D9 no day-boundary message (T6 default behaviour), D10 reflection fallback (T8 conditional).

All 9 risks have at least one task gate addressing them: R1 (T8 step 2-3), R2 (T3 step 3 — uses add_sync), R3 (T1+T7 — independent volume), R4 (T7 step 2), R5 (T8 step 3), R6 (T7 step 5 — explicit create_or_get), R7 (T8 step 2 SQL check), R8 (acceptable per spec), R9 (T7 step 5).

**Placeholder scan:** All code blocks contain runnable code or exact commands. Result-doc templates have `<placeholder>` markers but those are intentional (numbers come from the run output in Task 9).

**Type consistency:** `LegacyMirixClient.search_procedural` returns `list[dict]` everywhere it's referenced. `LegacyMirixEvolver.evolve_async` returns `{"sent": int}` everywhere. `legacy_procedural_to_metaclaw` returns `{name, description, content, category}` everywhere.

---

## Parallelisation hints (for subagent dispatch)

Tasks 2, 3, 4, 5 are independent (different files, no shared types beyond the imports each declares). They can be dispatched in parallel by the subagent-driven-development orchestrator. Task 6 depends on 3+4+5 being merged. Tasks 7-11 are strictly sequential.
