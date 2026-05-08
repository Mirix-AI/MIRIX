# MIRIX × MetaClaw Evolution-Bench Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-day end-to-end evaluation harness that swaps MetaClaw's
`SkillEvolver` and `SkillManager` for MIRIX's procedural memory subsystem,
runs `metaclaw-bench` day01..day03, and produces per-day pass-rate metrics
plus an evolution-trajectory summary.

**Architecture:** A driver in `evals/metaclaw/` clones MetaClaw, subclasses
its `SkillEvolver`/`SkillManager` to delegate to MIRIX's REST API
(`/v1/skills/evolve`, `/v1/skills?query=...`), runs a self-contained Qwen3-
native tool-call agent loop against `questions.json` in a copied workspace,
scores rounds via the bench's existing eval scripts, and triggers MIRIX's
`ProceduralMemoryAgent` to evolve at each day's end. Both the bench-side
agent and MIRIX's internal sub-agents share `openai/gpt-5.2` chat and
`google/gemini-embedding-001` embedding via OpenRouter (single API key,
single base URL).

**Tech Stack:** Python 3.10+, `httpx`, `openai>=1.0`, `pytest`, MIRIX (this
repo on `feat/skill-evolve`), `aiming-lab/MetaClaw` (cloned for class
inheritance only — not for its rollout), PostgreSQL with pgvector (already
running via existing `e2e-postgres` container), OpenRouter.

**Reference spec:** `docs/superpowers/specs/2026-05-08-mirix-metaclaw-eval-design.md`

---

## Pre-flight

Before Task 1, verify the working directory is `/Users/nanjiayan/Desktop/awesome_agent/MIRIX`
and the current branch is `feat/skill-evolve`. All paths in this plan are
relative to that directory unless an absolute path is given.

```bash
pwd                                    # expect: .../MIRIX
git branch --show-current              # expect: feat/skill-evolve
docker ps --format '{{.Names}}\t{{.Status}}' | grep e2e-postgres
                                       # expect: healthy
```

If `e2e-postgres` is not running, start it:

```bash
docker start e2e-postgres              # or: docker-compose up -d postgres
```

The MIRIX API server is **not** assumed to be running — Task 12 starts it.

---

## Task 1: Clone MetaClaw and pin its Python deps

**Files:**
- Create: `third_party/MetaClaw/` (cloned)
- Modify: `.gitignore` (add `third_party/`)
- Modify: `evals/metaclaw/README.md` (created in Task 2 — for now skip)

- [ ] **Step 1.1: Add `third_party/` to `.gitignore`**

Edit `.gitignore`, append at end of file:

```
# Third-party repos cloned for evals (cloned via plan tasks, not committed)
third_party/
```

- [ ] **Step 1.2: Clone MetaClaw**

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/aiming-lab/MetaClaw third_party/MetaClaw
```

Expected: directory `third_party/MetaClaw/` exists, has `pyproject.toml`,
`metaclaw/` subdir, `benchmark/` subdir.

Verify:

```bash
ls third_party/MetaClaw/metaclaw/skill_evolver.py
ls third_party/MetaClaw/metaclaw/skill_manager.py
ls third_party/MetaClaw/benchmark/data/metaclaw-bench/eval/day01/questions.json
```

All three files must exist.

- [ ] **Step 1.3: Install MetaClaw with `[evolve]` extras only**

```bash
pip install -e "./third_party/MetaClaw[evolve]"
```

If pip rejects extras (no extras defined), fall back to:

```bash
pip install -e ./third_party/MetaClaw
pip install openai>=1.0 httpx pytest pytest-asyncio
```

- [ ] **Step 1.4: Verify the two interface classes import**

```bash
python -c "from metaclaw.skill_evolver import SkillEvolver; print('OK SkillEvolver')"
python -c "from metaclaw.skill_manager import SkillManager; print('OK SkillManager')"
python -c "from metaclaw.data_formatter import ConversationSample; print('OK ConversationSample')"
```

Expected: three `OK ...` lines, no ImportError.

If `ConversationSample` import fails, find the actual location:

```bash
grep -rn "class ConversationSample" third_party/MetaClaw/metaclaw/
```

Note the actual import path; use it everywhere `ConversationSample` appears
later in this plan.

- [ ] **Step 1.5: Commit infrastructure changes**

```bash
git add .gitignore
git commit -m "[VEPEAGE-000] eval: gitignore third_party/ for MetaClaw clone"
```

---

## Task 2: Create the eval harness skeleton

**Files:**
- Create: `evals/metaclaw/__init__.py`
- Create: `evals/metaclaw/README.md`
- Create: `evals/metaclaw/tests/__init__.py`

- [ ] **Step 2.1: Create directory layout**

```bash
mkdir -p evals/metaclaw/tests
touch evals/metaclaw/__init__.py
touch evals/metaclaw/tests/__init__.py
```

- [ ] **Step 2.2: Write `evals/metaclaw/README.md`**

Create file with this exact content:

````markdown
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
````

- [ ] **Step 2.3: Commit skeleton**

```bash
git add evals/metaclaw/__init__.py evals/metaclaw/tests/__init__.py evals/metaclaw/README.md
git commit -m "[VEPEAGE-000] eval: scaffold evals/metaclaw/ harness skeleton"
```

---

## Task 3: `format_adapter.py` — skill and round serialization (TDD)

**Files:**
- Create: `evals/metaclaw/tests/test_format_adapter.py`
- Create: `evals/metaclaw/format_adapter.py`

- [ ] **Step 3.1: Write failing test for `mirix_to_metaclaw`**

Create `evals/metaclaw/tests/test_format_adapter.py`:

```python
"""Tests for evals.metaclaw.format_adapter."""
from evals.metaclaw.format_adapter import (
    mirix_to_metaclaw,
    round_to_message,
    RoundResult,
)


def test_mirix_to_metaclaw_maps_required_fields():
    mirix_skill = {
        "id": "proc-abc",
        "name": "iso8601-with-cst-offset",
        "description": "Format datetimes as YYYY-MM-DDTHH:MM:SS+08:00.",
        "instructions": "When asked to record any datetime field...",
        "entry_type": "guide",
        "version": "0.1.0",
    }
    out = mirix_to_metaclaw(mirix_skill)
    assert out["name"] == "iso8601-with-cst-offset"
    assert out["description"].startswith("Format datetimes")
    assert out["content"].startswith("When asked")
    assert out["category"] == "guide"


def test_mirix_to_metaclaw_defaults_category_when_missing():
    out = mirix_to_metaclaw({
        "name": "x", "description": "d", "instructions": "i"
    })
    assert out["category"] == "general"
```

- [ ] **Step 3.2: Run the test, verify it fails**

```bash
pytest evals/metaclaw/tests/test_format_adapter.py::test_mirix_to_metaclaw_maps_required_fields -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evals.metaclaw.format_adapter'`.

- [ ] **Step 3.3: Write minimal `mirix_to_metaclaw`**

Create `evals/metaclaw/format_adapter.py`:

```python
"""Schema bridge between MIRIX skill objects and MetaClaw skill dicts,
plus serialization of one bench round into a single evolve-message string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundResult:
    """One bench round outcome — what we send to MIRIX evolve."""
    round_id: str
    round_type: str            # "file_check" | "multi_choice"
    question: str
    final_answer: str          # agent's final tool output or bbox answer
    reward: float              # 1.0 pass, 0.0 fail
    eval_outcome: str          # "pass" | "fail"
    feedback: str              # bench's correct/incorrect feedback string
    transcript: list[dict] = field(default_factory=list)
    error: str | None = None


def mirix_to_metaclaw(skill: dict[str, Any]) -> dict[str, Any]:
    """MIRIX skill dict (id/name/description/instructions/entry_type/version)
    → MetaClaw skill dict (name/description/content/category)."""
    return {
        "name":        skill["name"],
        "description": skill["description"],
        "content":     skill["instructions"],
        "category":    skill.get("entry_type") or "general",
    }


def round_to_message(r: RoundResult) -> str:
    """Serialize one round into a single evolve-message string."""
    raise NotImplementedError("Implemented in Step 3.5")
```

- [ ] **Step 3.4: Run tests, verify they pass**

```bash
pytest evals/metaclaw/tests/test_format_adapter.py::test_mirix_to_metaclaw_maps_required_fields evals/metaclaw/tests/test_format_adapter.py::test_mirix_to_metaclaw_defaults_category_when_missing -v
```

Expected: 2 passed.

- [ ] **Step 3.5: Write failing test for `round_to_message`**

Append to `evals/metaclaw/tests/test_format_adapter.py`:

```python
def test_round_to_message_includes_outcome_and_feedback():
    r = RoundResult(
        round_id="r1",
        round_type="file_check",
        question="Save standup notes...",
        final_answer="Wrote day01/standup.json",
        reward=0.0,
        eval_outcome="fail",
        feedback="Time fields must use full datetime with +08:00 offset.",
    )
    msg = round_to_message(r)
    assert "r1" in msg
    assert "FAIL" in msg or "fail" in msg.lower()
    assert "+08:00" in msg              # feedback content carried through
    assert "Save standup notes" in msg


def test_round_to_message_handles_multichoice():
    r = RoundResult(
        round_id="r3",
        round_type="multi_choice",
        question="Which formats are valid?",
        final_answer="\\bbox{A,E}",
        reward=1.0,
        eval_outcome="pass",
        feedback="Correct! A and E both valid.",
    )
    msg = round_to_message(r)
    assert "PASS" in msg or "pass" in msg.lower()
    assert "\\bbox{A,E}" in msg
```

- [ ] **Step 3.6: Run, verify FAIL**

```bash
pytest evals/metaclaw/tests/test_format_adapter.py -v
```

Expected: 2 passed, 2 failed (the two new tests).

- [ ] **Step 3.7: Implement `round_to_message`**

Replace the `NotImplementedError` body in `evals/metaclaw/format_adapter.py`:

```python
def round_to_message(r: RoundResult) -> str:
    status = "PASS" if r.reward >= 1.0 else "FAIL"
    parts = [
        f"### Round {r.round_id}  [{r.round_type}]  outcome={status}",
        "",
        "**Question:**",
        r.question.strip(),
        "",
        "**Agent final answer:**",
        r.final_answer.strip() or "(empty)",
        "",
        "**Bench feedback:**",
        r.feedback.strip() or "(no feedback)",
    ]
    if r.error:
        parts += ["", f"**Error:** {r.error}"]
    return "\n".join(parts)
```

- [ ] **Step 3.8: Run all tests, verify pass**

```bash
pytest evals/metaclaw/tests/test_format_adapter.py -v
```

Expected: 4 passed.

- [ ] **Step 3.9: Commit**

```bash
git add evals/metaclaw/format_adapter.py evals/metaclaw/tests/test_format_adapter.py
git commit -m "[VEPEAGE-000] eval: add format adapter (skill schema + round serialization)"
```

---

## Task 4: `llm_config_helpers.py` — build OpenRouter LLMConfig

**Files:**
- Create: `evals/metaclaw/llm_config_helpers.py`

This module is consumed by Task 12 (MIRIX server config). No tests — it's
plumbing that builds dataclass instances; correctness is verified at
runtime in Task 12.

- [ ] **Step 4.1: Write the helper**

Create `evals/metaclaw/llm_config_helpers.py`:

```python
"""LLMConfig / EmbeddingConfig builders for the OpenRouter setup used by
this eval harness. Both chat and embedding go through OpenRouter; the
MIRIX OpenAI client (`mirix/llm_api/openai_client.py`) accepts arbitrary
`base_url`, so the same client class serves both endpoints.
"""
from __future__ import annotations

import os

from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-5.2"
DEFAULT_EMBED_MODEL = "google/gemini-embedding-001"
DEFAULT_EMBED_DIM = 1536


def openrouter_chat_config(model: str | None = None) -> LLMConfig:
    return LLMConfig(
        model=model or os.environ.get("EVAL_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        model_endpoint_type="openai",
        model_endpoint=OPENROUTER_BASE_URL,
        context_window=128_000,
    )


def openrouter_embedding_config(
    model: str | None = None,
    dim: int | None = None,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        embedding_model=model or os.environ.get("EVAL_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embedding_endpoint_type="openai",
        embedding_endpoint=OPENROUTER_BASE_URL,
        embedding_dim=int(dim or os.environ.get("EVAL_EMBED_DIM", DEFAULT_EMBED_DIM)),
        embedding_chunk_size=300,
    )


def assert_openrouter_env() -> None:
    """Fail fast if env is not configured."""
    missing = [k for k in ("OPENAI_API_KEY",) if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing env vars: {missing}. "
            f"Set OPENAI_API_KEY to your OpenRouter key, "
            f"OPENAI_API_BASE to {OPENROUTER_BASE_URL}."
        )
```

- [ ] **Step 4.2: Sanity-check imports**

```bash
python -c "from evals.metaclaw.llm_config_helpers import openrouter_chat_config; print(openrouter_chat_config())"
```

Expected: prints an `LLMConfig(model='openai/gpt-5.2', model_endpoint_type='openai', ...)` representation. If `EmbeddingConfig` import fails, find the right module:

```bash
grep -rn "class EmbeddingConfig" mirix/schemas/
```

Update the import in the helper accordingly.

- [ ] **Step 4.3: Commit**

```bash
git add evals/metaclaw/llm_config_helpers.py
git commit -m "[VEPEAGE-000] eval: add OpenRouter LLMConfig/EmbeddingConfig helpers"
```

---

## Task 5: `mirix_client.py` — httpx wrapper around MIRIX REST (TDD)

**Files:**
- Create: `evals/metaclaw/tests/test_mirix_client.py`
- Create: `evals/metaclaw/mirix_client.py`

- [ ] **Step 5.1: Write failing test for `evolve` request shape**

Create `evals/metaclaw/tests/test_mirix_client.py`:

```python
"""Tests for evals.metaclaw.mirix_client."""
import json

import httpx
import pytest

from evals.metaclaw.mirix_client import MirixClient


@pytest.mark.asyncio
async def test_evolve_posts_messages_and_user_id():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content.decode())
        return httpx.Response(
            200,
            json={"success": True,
                  "changes": {
                      "created": [{"id": "proc-1", "name": "iso8601",
                                   "description": "d", "instructions": "i",
                                   "entry_type": "guide", "version": "0.1.0"}],
                      "edited": [], "deleted": []}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="user-1", _client=ac)
        result = await client.evolve(["msg1", "msg2"])

    assert seen["url"].endswith("/v1/skills/evolve")
    assert seen["body"] == {"messages": ["msg1", "msg2"], "user_id": "user-1"}
    assert result["created"][0]["name"] == "iso8601"
    assert result["edited"] == []
    assert result["deleted"] == []


@pytest.mark.asyncio
async def test_search_skills_calls_bm25_and_returns_skills():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(
            200,
            json={"skills": [{"id": "proc-1", "name": "iso8601",
                              "description": "d", "instructions": "i",
                              "entry_type": "guide", "version": "0.1.0"}],
                  "total_count": 1},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="user-1", _client=ac)
        skills = await client.search_skills("datetime format", limit=3)

    assert "/v1/skills" in seen["url"]
    assert "search_method=bm25" in seen["url"]
    assert "query=datetime+format" in seen["url"] or "query=datetime%20format" in seen["url"]
    assert "limit=3" in seen["url"]
    assert skills[0]["name"] == "iso8601"


@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"}))
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="u", _client=ac)
        assert await client.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_on_500():
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as ac:
        client = MirixClient(base_url="http://x", user_id="u", _client=ac)
        assert await client.health() is False
```

- [ ] **Step 5.2: Run, verify FAIL**

```bash
pytest evals/metaclaw/tests/test_mirix_client.py -v
```

Expected: ModuleNotFoundError for `evals.metaclaw.mirix_client`.

- [ ] **Step 5.3: Implement `mirix_client.py`**

Create `evals/metaclaw/mirix_client.py`:

```python
"""Thin async wrapper around MIRIX REST endpoints used by this eval harness.

Endpoints used:
    POST /v1/skills/evolve   — trigger ProceduralMemoryAgent on a batch of
                               round messages, returns created/edited/deleted diff
    GET  /v1/skills?...      — search skills (BM25); used for retrieval
    GET  /healthz / /        — liveness probe
"""
from __future__ import annotations

from typing import Any

import httpx


class MirixClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8531",
        user_id: str = "eval-metaclaw-3day",
        timeout: float = 600.0,
        _client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self._timeout = timeout
        self._client = _client            # injectable for tests
        self._owns_client = _client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self._timeout
            )
        return self._client

    async def aclose(self):
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def evolve(self, messages: list[str]) -> dict[str, Any]:
        """POST /v1/skills/evolve. Returns {created: [...], edited: [...], deleted: [...]}."""
        client = await self._get_client()
        resp = await client.post(
            "/v1/skills/evolve",
            json={"messages": messages, "user_id": self.user_id},
        )
        resp.raise_for_status()
        body = resp.json()
        # rest_api.py returns {success, changes: {created, edited, deleted}}
        return body.get("changes", body)

    async def search_skills(
        self,
        query: str,
        limit: int = 6,
        search_method: str = "bm25",
        search_field: str = "description",
    ) -> list[dict[str, Any]]:
        """GET /v1/skills?query=...&search_method=bm25&search_field=description&limit=N."""
        client = await self._get_client()
        resp = await client.get(
            "/v1/skills",
            params={
                "query": query,
                "limit": limit,
                "search_method": search_method,
                "search_field": search_field,
                "user_id": self.user_id,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("skills", [])

    async def health(self) -> bool:
        client = await self._get_client()
        try:
            resp = await client.get("/")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 5.4: Add `pytest-asyncio` config**

If `evals/metaclaw/tests/conftest.py` does not exist, create:

```python
# evals/metaclaw/tests/conftest.py
import pytest_asyncio  # noqa: F401   ensure plugin loads

# Use asyncio mode for all tests in this dir
import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "asyncio" in item.keywords or item.get_closest_marker("asyncio"):
            continue
```

Or simpler: append `[tool.pytest.ini_options]` `asyncio_mode = "auto"` to root `pyproject.toml` if not already present. Verify:

```bash
grep -n "asyncio_mode" pyproject.toml
```

If missing, append (under existing `[tool.pytest.ini_options]` section, or create one):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 5.5: Run all `mirix_client` tests, verify pass**

```bash
pytest evals/metaclaw/tests/test_mirix_client.py -v
```

Expected: 4 passed.

- [ ] **Step 5.6: Commit**

```bash
git add evals/metaclaw/mirix_client.py evals/metaclaw/tests/test_mirix_client.py evals/metaclaw/tests/conftest.py pyproject.toml
git commit -m "[VEPEAGE-000] eval: add async MirixClient (evolve + search + health)"
```

---

## Task 6: `mirix_skill_evolver.py` — subclass MetaClaw `SkillEvolver`

**Files:**
- Create: `evals/metaclaw/tests/test_mirix_skill_evolver.py`
- Create: `evals/metaclaw/mirix_skill_evolver.py`

- [ ] **Step 6.1: Write failing test**

Create `evals/metaclaw/tests/test_mirix_skill_evolver.py`:

```python
"""Tests for evals.metaclaw.mirix_skill_evolver."""
from unittest.mock import AsyncMock

import pytest

from evals.metaclaw.format_adapter import RoundResult
from evals.metaclaw.mirix_skill_evolver import MirixSkillEvolver


@pytest.mark.asyncio
async def test_evolve_serializes_rounds_to_messages_and_returns_metaclaw_skills():
    mirix = AsyncMock()
    mirix.evolve.return_value = {
        "created": [
            {"name": "iso8601", "description": "d", "instructions": "i",
             "entry_type": "guide", "version": "0.1.0"},
        ],
        "edited": [],
        "deleted": [],
    }

    evolver = MirixSkillEvolver(mirix_client=mirix)

    rounds = [
        RoundResult("r1", "file_check", "Q1", "A1", 0.0, "fail", "FB1"),
        RoundResult("r2", "multi_choice", "Q2", "\\bbox{A}", 1.0, "pass", "FB2"),
    ]

    out = await evolver.evolve(rounds, current_skills={})

    # Serialization: 2 rounds → 2 messages
    args, _ = mirix.evolve.call_args
    sent_messages = args[0]
    assert len(sent_messages) == 2
    assert "r1" in sent_messages[0]
    assert "r2" in sent_messages[1]

    # Output: metaclaw-shaped skills
    assert out == [
        {"name": "iso8601", "description": "d", "content": "i", "category": "guide"}
    ]


def test_should_evolve_always_true_to_let_driver_decide():
    evolver = MirixSkillEvolver(mirix_client=None)
    assert evolver.should_evolve([]) is True
    assert evolver.should_evolve([1, 2, 3], threshold=0.99) is True
```

- [ ] **Step 6.2: Run, verify FAIL**

```bash
pytest evals/metaclaw/tests/test_mirix_skill_evolver.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 6.3: Implement `mirix_skill_evolver.py`**

Create `evals/metaclaw/mirix_skill_evolver.py`:

```python
"""MIRIX-backed replacement for MetaClaw's SkillEvolver.

The evolve trigger logic (success-rate threshold) is bypassed: the driver
calls evolve() exactly once per day at end-of-day. Within evolve(), each
RoundResult is serialized to one MIRIX evolve-message (success and failure
both included), MIRIX's ProceduralMemoryAgent decides what to create / edit
/ delete, and the resulting skills are returned in MetaClaw's
{name, description, content, category} format.
"""
from __future__ import annotations

from typing import Any, Iterable

from metaclaw.skill_evolver import SkillEvolver

from evals.metaclaw.format_adapter import (
    RoundResult,
    mirix_to_metaclaw,
    round_to_message,
)
from evals.metaclaw.mirix_client import MirixClient


class MirixSkillEvolver(SkillEvolver):
    """Subclass that delegates evolve() to MIRIX's REST endpoint."""

    def __init__(self, mirix_client: MirixClient | None, **kwargs: Any) -> None:
        # Bypass parent __init__: it expects OpenAI env vars we don't need
        # here because the LLM call happens inside MIRIX, not in this class.
        self.mirix = mirix_client
        self.update_history: list[dict] = []
        self.history_path = None

    def should_evolve(self, batch, threshold: float = 0.0) -> bool:  # noqa: ARG002
        # Driver-driven: always allow; the driver decides timing.
        return True

    async def evolve(
        self,
        failed_samples: Iterable[RoundResult],
        current_skills: dict | None = None,    # noqa: ARG002 (signature parity)
    ) -> list[dict]:
        rounds = list(failed_samples)
        if not rounds:
            return []
        messages = [round_to_message(r) for r in rounds]
        diff = await self.mirix.evolve(messages)
        produced = list(diff.get("created", [])) + list(diff.get("edited", []))
        return [mirix_to_metaclaw(s) for s in produced]
```

- [ ] **Step 6.4: Run tests, verify pass**

```bash
pytest evals/metaclaw/tests/test_mirix_skill_evolver.py -v
```

Expected: 2 passed.

- [ ] **Step 6.5: Commit**

```bash
git add evals/metaclaw/mirix_skill_evolver.py evals/metaclaw/tests/test_mirix_skill_evolver.py
git commit -m "[VEPEAGE-000] eval: add MirixSkillEvolver (subclass delegating to MIRIX REST)"
```

---

## Task 7: `mirix_skill_manager.py` — subclass MetaClaw `SkillManager`

**Files:**
- Create: `evals/metaclaw/tests/test_mirix_skill_manager.py`
- Create: `evals/metaclaw/mirix_skill_manager.py`

The parent's `__init__` scans a directory for `SKILL.md` files. Our subclass
skips that scan entirely; skills always come from MIRIX. Retrieval is sync
in the parent's interface, so we wrap the async client in `asyncio.run`.

- [ ] **Step 7.1: Write failing test**

Create `evals/metaclaw/tests/test_mirix_skill_manager.py`:

```python
"""Tests for evals.metaclaw.mirix_skill_manager."""
from unittest.mock import AsyncMock, MagicMock

from evals.metaclaw.mirix_skill_manager import MirixSkillManager


def test_retrieve_returns_metaclaw_shaped_skills():
    mirix = MagicMock()
    mirix.search_skills = AsyncMock(return_value=[
        {"name": "iso8601", "description": "d", "instructions": "i",
         "entry_type": "guide", "version": "0.1.0"},
    ])

    mgr = MirixSkillManager(mirix_client=mirix)

    out = mgr.retrieve("datetime format please", top_k=3)

    mirix.search_skills.assert_called_once()
    call_kwargs = mirix.search_skills.call_args.kwargs
    assert call_kwargs.get("limit") == 3
    assert out == [
        {"name": "iso8601", "description": "d", "content": "i", "category": "guide"}
    ]


def test_retrieve_with_no_skills_returns_empty():
    mirix = MagicMock()
    mirix.search_skills = AsyncMock(return_value=[])

    mgr = MirixSkillManager(mirix_client=mirix)
    assert mgr.retrieve("anything") == []
```

- [ ] **Step 7.2: Run, verify FAIL**

```bash
pytest evals/metaclaw/tests/test_mirix_skill_manager.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 7.3: Implement `mirix_skill_manager.py`**

Create `evals/metaclaw/mirix_skill_manager.py`:

```python
"""MIRIX-backed replacement for MetaClaw's SkillManager.

The parent class scans a directory of SKILL.md files; this subclass bypasses
that scan and retrieves skills from MIRIX (BM25 over description) every time
retrieve() is called. The parent's retrieval interface is sync, so we wrap
the async MIRIX client via asyncio. Public API parity is enough for the
metaclaw-side code paths that consume `.retrieve()`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from metaclaw.skill_manager import SkillManager

from evals.metaclaw.format_adapter import mirix_to_metaclaw
from evals.metaclaw.mirix_client import MirixClient


class MirixSkillManager(SkillManager):
    """Subclass whose retrieve() delegates to MIRIX REST."""

    def __init__(self, mirix_client: MirixClient, **kwargs: Any) -> None:
        # Bypass parent __init__ (which requires a skills_dir to scan).
        self.mirix = mirix_client
        self.skills: dict[str, Any] = {
            "general_skills": [], "task_specific_skills": {}, "common_mistakes": []
        }
        self.generation: int = 0

    def retrieve(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        skills = _run_sync(self.mirix.search_skills(query=query, limit=top_k))
        return [mirix_to_metaclaw(s) for s in skills]


def _run_sync(coro):
    """Run an async coroutine from a sync caller. Works whether or not an
    event loop is already running in the current thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We are inside an async context already — bounce onto a fresh loop
        # in a thread. Acceptable here because retrieve() is on the bench
        # path, not on a tight inner loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
```

- [ ] **Step 7.4: Run tests, verify pass**

```bash
pytest evals/metaclaw/tests/test_mirix_skill_manager.py -v
```

Expected: 2 passed.

- [ ] **Step 7.5: Commit**

```bash
git add evals/metaclaw/mirix_skill_manager.py evals/metaclaw/tests/test_mirix_skill_manager.py
git commit -m "[VEPEAGE-000] eval: add MirixSkillManager (subclass delegating to MIRIX REST)"
```

---

## Task 8: `round_runner.py` — Qwen3-native agent loop + scoring (TDD)

**Files:**
- Create: `evals/metaclaw/tests/test_round_runner.py`
- Create: `evals/metaclaw/round_runner.py`

This is the largest single component. It implements:

1. A tool-calling agent loop using OpenAI's `chat.completions` with
   function-calling tools (`bash`, `read_file`, `write_file`, `list_dir`).
   Tools execute against a working directory bound at run start. The loop
   terminates when the model emits no more tool calls or hits `max_turns`.
2. Two scorers:
   - `file_check`: `subprocess.run(eval.command, cwd=workspace)`,
     `reward = 1.0 if returncode == 0 else 0.0`.
   - `multi_choice`: parse `\bbox{A,E}` from the model's last assistant
     message, compare to `eval.answer` (set equality, ignore order).

We split into two test files conceptually but keep them in one for
brevity: scorer tests (no LLM needed) and a loop-orchestration test
(LLM mocked).

- [ ] **Step 8.1: Write failing tests for the scorers**

Create `evals/metaclaw/tests/test_round_runner.py`:

```python
"""Tests for evals.metaclaw.round_runner."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evals.metaclaw.round_runner import (
    parse_bbox_answer,
    score_file_check,
    score_multi_choice,
)


def test_parse_bbox_simple():
    assert parse_bbox_answer("Final: \\bbox{A,E}") == ["A", "E"]


def test_parse_bbox_single_letter():
    assert parse_bbox_answer("\\bbox{B}") == ["B"]


def test_parse_bbox_with_spaces():
    assert parse_bbox_answer("answer \\bbox{ A , C , F } done") == ["A", "C", "F"]


def test_parse_bbox_missing_returns_empty():
    assert parse_bbox_answer("nope, no answer here") == []


def test_score_multi_choice_correct_set_equality_ignores_order():
    eval_block = {"answer": ["A", "E"]}
    assert score_multi_choice("\\bbox{E,A}", eval_block) == (1.0, "pass")


def test_score_multi_choice_wrong_subset():
    eval_block = {"answer": ["A", "E"]}
    assert score_multi_choice("\\bbox{A}", eval_block) == (0.0, "fail")


def test_score_multi_choice_wrong_extra():
    eval_block = {"answer": ["A", "E"]}
    assert score_multi_choice("\\bbox{A,E,F}", eval_block) == (0.0, "fail")


def test_score_file_check_passes_when_command_exits_0(tmp_path: Path):
    eval_block = {
        "command": "true",      # POSIX: exits 0
        "expect_exit": 0,
    }
    reward, outcome = score_file_check(eval_block, workspace=tmp_path)
    assert (reward, outcome) == (1.0, "pass")


def test_score_file_check_fails_when_command_exits_nonzero(tmp_path: Path):
    eval_block = {"command": "false", "expect_exit": 0}
    reward, outcome = score_file_check(eval_block, workspace=tmp_path)
    assert (reward, outcome) == (0.0, "fail")
```

- [ ] **Step 8.2: Run, verify FAIL**

```bash
pytest evals/metaclaw/tests/test_round_runner.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 8.3: Implement scorers + tool layer + loop in `round_runner.py`**

Create `evals/metaclaw/round_runner.py`:

```python
"""Single-round agent loop and scoring for the MetaClaw bench.

Tools follow the OpenAI function-calling schema:
  - bash(command):       run shell command in the round's workspace
  - read_file(path):     read text file (UTF-8, max 100KB)
  - write_file(path, content):  overwrite file with content
  - list_dir(path):      list directory entries

The loop terminates when:
  - the assistant message has no tool_calls (final answer reached), or
  - max_turns (default 20) is hit, or
  - wallclock cap (default 300 s) is hit.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evals.metaclaw.format_adapter import RoundResult


_BBOX_RE = re.compile(r"\\bbox\{([^}]+)\}")


def parse_bbox_answer(text: str) -> list[str]:
    """Extract letters from the LAST occurrence of \\bbox{...}."""
    matches = _BBOX_RE.findall(text or "")
    if not matches:
        return []
    inner = matches[-1]
    letters = [s.strip().upper() for s in inner.split(",") if s.strip()]
    return letters


def score_multi_choice(final_answer: str, eval_block: dict) -> tuple[float, str]:
    expected = set(eval_block.get("answer", []))
    got = set(parse_bbox_answer(final_answer))
    return (1.0, "pass") if got == expected else (0.0, "fail")


def score_file_check(eval_block: dict, workspace: Path) -> tuple[float, str]:
    cmd = eval_block.get("command", "")
    expect = int(eval_block.get("expect_exit", 0))
    if not cmd:
        return (0.0, "fail")
    proc = subprocess.run(
        cmd, shell=True, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    return (1.0, "pass") if proc.returncode == expect else (0.0, "fail")


# -- Tools -------------------------------------------------------------------

def _tool_bash(workspace: Path, command: str) -> str:
    proc = subprocess.run(
        command, shell=True, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    out = (proc.stdout or "")[-4000:]
    err = (proc.stderr or "")[-2000:]
    return f"exit={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"


def _tool_read_file(workspace: Path, path: str) -> str:
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        return "ERROR: path escapes workspace"
    if not p.exists():
        return f"ERROR: not found: {path}"
    data = p.read_bytes()[:102_400]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _tool_write_file(workspace: Path, path: str, content: str) -> str:
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        return "ERROR: path escapes workspace"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


def _tool_list_dir(workspace: Path, path: str = ".") -> str:
    p = (workspace / path).resolve()
    if not str(p).startswith(str(workspace.resolve())):
        return "ERROR: path escapes workspace"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    items = []
    for entry in sorted(p.iterdir()):
        kind = "d" if entry.is_dir() else "f"
        items.append(f"{kind}\t{entry.relative_to(workspace)}")
    return "\n".join(items) or "(empty)"


_TOOLS_SCHEMA: list[dict] = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file relative to the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write (overwrite) a text file relative to the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List entries in a directory relative to the workspace.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string", "default": "."}},
            "required": [],
        }}},
]


def _dispatch_tool(name: str, args: dict, workspace: Path) -> str:
    if name == "bash":
        return _tool_bash(workspace, args.get("command", ""))
    if name == "read_file":
        return _tool_read_file(workspace, args.get("path", ""))
    if name == "write_file":
        return _tool_write_file(workspace, args.get("path", ""), args.get("content", ""))
    if name == "list_dir":
        return _tool_list_dir(workspace, args.get("path", "."))
    return f"ERROR: unknown tool {name}"


# -- Loop --------------------------------------------------------------------

@dataclass
class RunnerConfig:
    chat_model: str
    workspace: Path
    max_turns: int = 20
    wallclock_cap_s: float = 300.0


SYSTEM_PROMPT_BASE = (
    "You are an agent solving a single task. The user will give you ONE "
    "question. Use the provided tools (bash, read_file, write_file, "
    "list_dir) to inspect the workspace and produce the requested output. "
    "When the task is complete, reply with a brief final message and STOP "
    "calling tools. For multiple-choice questions, end your final message "
    "with \\bbox{X} or \\bbox{X,Y}."
)


def build_system_prompt(skills: list[dict]) -> str:
    if not skills:
        return SYSTEM_PROMPT_BASE
    parts = [SYSTEM_PROMPT_BASE, "", "## Relevant skills"]
    for s in skills:
        parts.append(f"### {s['name']}  ({s.get('category','general')})")
        parts.append(s.get("description", "").strip())
        parts.append("")
        parts.append(s.get("content", "").strip())
        parts.append("")
    return "\n".join(parts)


def run_round(
    *,
    openai_client,
    cfg: RunnerConfig,
    round_id: str,
    round_type: str,
    question: str,
    eval_block: dict,
    skills: list[dict],
) -> RoundResult:
    system = build_system_prompt(skills)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    transcript: list[dict] = []
    started = time.monotonic()
    final_text = ""
    error: str | None = None

    for turn in range(cfg.max_turns):
        if time.monotonic() - started > cfg.wallclock_cap_s:
            error = "wallclock_cap"
            break
        resp = openai_client.chat.completions.create(
            model=cfg.chat_model,
            messages=messages,
            tools=_TOOLS_SCHEMA,
            tool_choice="auto",
        )
        choice = resp.choices[0]
        msg = choice.message
        transcript.append({"role": "assistant", "content": msg.content,
                           "tool_calls": [
                               {"name": tc.function.name,
                                "arguments": tc.function.arguments}
                               for tc in (msg.tool_calls or [])
                           ]})
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ],
        })
        if not msg.tool_calls:
            final_text = msg.content or ""
            break
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(tc.function.name, args, cfg.workspace)
            transcript.append({"role": "tool", "name": tc.function.name,
                               "result": result[:1000]})
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })
    else:
        error = "turn_limit"

    # Score
    if round_type == "multi_choice":
        reward, outcome = score_multi_choice(final_text, eval_block)
    elif round_type == "file_check":
        reward, outcome = score_file_check(eval_block, cfg.workspace)
    else:
        reward, outcome = (0.0, "fail")
        error = error or f"unknown_round_type:{round_type}"

    return RoundResult(
        round_id=round_id, round_type=round_type, question=question,
        final_answer=final_text, reward=reward, eval_outcome=outcome,
        feedback="",   # filled in by driver from questions.json
        transcript=transcript, error=error,
    )
```

- [ ] **Step 8.4: Run scorer tests, verify pass**

```bash
pytest evals/metaclaw/tests/test_round_runner.py -v
```

Expected: 9 passed.

- [ ] **Step 8.5: Add a loop-orchestration test with mocked LLM**

Append to `evals/metaclaw/tests/test_round_runner.py`:

```python
def _make_fake_openai(responses: list[dict]):
    """Build a fake OpenAI client whose chat.completions.create returns
    the given responses in order. Each entry is either:
        {"text": str, "tool_calls": [{"id":..., "name":..., "args":{...}}]}
    """
    from types import SimpleNamespace

    class Calls:
        def __init__(self):
            self._idx = 0
        def create(self, **kwargs):
            r = responses[self._idx]
            self._idx += 1
            tcs = []
            for i, tc in enumerate(r.get("tool_calls", []) or []):
                tcs.append(SimpleNamespace(
                    id=tc.get("id", f"call-{i}"),
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=json.dumps(tc.get("args", {})),
                    ),
                ))
            choice = SimpleNamespace(message=SimpleNamespace(
                content=r.get("text"), tool_calls=tcs or None,
            ))
            return SimpleNamespace(choices=[choice])

    chat = SimpleNamespace(completions=Calls())
    return SimpleNamespace(chat=chat)


def test_run_round_multi_choice_pass(tmp_path: Path):
    from evals.metaclaw.round_runner import RunnerConfig, run_round
    fake = _make_fake_openai([
        {"text": "Answer: \\bbox{A,E}"}                    # no tool calls → terminate
    ])
    res = run_round(
        openai_client=fake,
        cfg=RunnerConfig(chat_model="x", workspace=tmp_path, max_turns=3),
        round_id="r3", round_type="multi_choice",
        question="Q?", eval_block={"answer": ["A", "E"]},
        skills=[],
    )
    assert res.reward == 1.0
    assert res.eval_outcome == "pass"
    assert res.final_answer.endswith("\\bbox{A,E}")


def test_run_round_file_check_writes_then_passes(tmp_path: Path):
    from evals.metaclaw.round_runner import RunnerConfig, run_round
    fake = _make_fake_openai([
        # turn 0: tool call to write a file
        {"text": None, "tool_calls": [
            {"id": "c0", "name": "write_file",
             "args": {"path": "out.txt", "content": "hello"}},
        ]},
        # turn 1: no tool calls → terminate
        {"text": "Done."},
    ])
    res = run_round(
        openai_client=fake,
        cfg=RunnerConfig(chat_model="x", workspace=tmp_path, max_turns=3),
        round_id="r1", round_type="file_check",
        question="Write hello to out.txt",
        eval_block={"command": "test -f out.txt && grep -q hello out.txt",
                    "expect_exit": 0},
        skills=[],
    )
    assert res.reward == 1.0
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_run_round_turn_limit_marks_error(tmp_path: Path):
    from evals.metaclaw.round_runner import RunnerConfig, run_round
    # Always emit one tool call → the loop never terminates naturally
    fake = _make_fake_openai([
        {"text": None, "tool_calls": [
            {"id": f"c{i}", "name": "list_dir", "args": {"path": "."}}
        ]} for i in range(10)
    ])
    res = run_round(
        openai_client=fake,
        cfg=RunnerConfig(chat_model="x", workspace=tmp_path, max_turns=2),
        round_id="r9", round_type="file_check",
        question="?",
        eval_block={"command": "false", "expect_exit": 0},
        skills=[],
    )
    assert res.error == "turn_limit"
    assert res.reward == 0.0
```

- [ ] **Step 8.6: Run the new tests, verify pass**

```bash
pytest evals/metaclaw/tests/test_round_runner.py -v
```

Expected: 12 passed.

- [ ] **Step 8.7: Commit**

```bash
git add evals/metaclaw/round_runner.py evals/metaclaw/tests/test_round_runner.py
git commit -m "[VEPEAGE-000] eval: add round_runner (Qwen3-style tool loop + scorers)"
```

---

## Task 9: `run_3day_eval.py` — driver

**Files:**
- Create: `evals/metaclaw/run_3day_eval.py`

This is procedural glue around the previously-built and tested components.
We add one focused integration test (`--dry-run` mode) instead of unit-
testing every helper.

- [ ] **Step 9.1: Write the driver**

Create `evals/metaclaw/run_3day_eval.py`:

```python
"""Driver: run metaclaw-bench day01..day03 against MIRIX as evolver+retriever.

Run from the MIRIX repo root:

    python -m evals.metaclaw.run_3day_eval                       # full 3 days
    python -m evals.metaclaw.run_3day_eval --days day01          # just day01
    python -m evals.metaclaw.run_3day_eval --max-rounds 1        # smoke
    python -m evals.metaclaw.run_3day_eval --dry-run             # no LLM, no MIRIX
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

logger = logging.getLogger("evals.metaclaw")

REPO_ROOT = Path(__file__).resolve().parents[2]
METACLAW_BENCH = REPO_ROOT / "third_party" / "MetaClaw" / "benchmark" / "data" / "metaclaw-bench"
EVAL_DIR = METACLAW_BENCH / "eval"
WORKSPACE_SRC = METACLAW_BENCH / "workspaces" / "shared"
SCORE_SCRIPT_DIR = REPO_ROOT / "third_party" / "MetaClaw" / "scripts"
DEFAULT_DAYS = ["day01", "day02", "day03"]


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _prepare_workspace(run_dir: Path) -> Path:
    """Copy workspaces/shared/ into run_dir/workspace/. Carries across all days."""
    ws = run_dir / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(WORKSPACE_SRC, ws)
    # Also copy bench scripts so eval.command lines like
    # `python scripts/check_iso8601.py` resolve from the workspace cwd.
    scripts_dst = ws / "scripts"
    if not scripts_dst.exists() and SCORE_SCRIPT_DIR.exists():
        shutil.copytree(SCORE_SCRIPT_DIR, scripts_dst)
    return ws


def _load_questions(day: str) -> dict:
    return json.loads((EVAL_DIR / day / "questions.json").read_text())


def _expected_feedback(round_obj: dict, outcome: str) -> str:
    fb = round_obj.get("feedback", {})
    if isinstance(fb, dict):
        return fb.get("correct", "") if outcome == "pass" else fb.get("incorrect", "")
    return str(fb)


async def _amain(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)

    from evals.metaclaw.format_adapter import RoundResult
    from evals.metaclaw.llm_config_helpers import (
        DEFAULT_CHAT_MODEL,
        OPENROUTER_BASE_URL,
        assert_openrouter_env,
    )
    from evals.metaclaw.mirix_client import MirixClient
    from evals.metaclaw.mirix_skill_evolver import MirixSkillEvolver
    from evals.metaclaw.mirix_skill_manager import MirixSkillManager
    from evals.metaclaw.round_runner import RunnerConfig, run_round

    if not args.dry_run:
        assert_openrouter_env()

    days = args.days or DEFAULT_DAYS
    run_id = _run_id()
    run_dir = REPO_ROOT / "evals" / "metaclaw" / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run %s — output dir %s", run_id, run_dir)

    workspace = _prepare_workspace(run_dir)
    logger.info("Workspace prepared at %s", workspace)

    # MIRIX wiring
    mirix = None
    evolver = None
    skill_mgr = None
    if not args.dry_run:
        mirix = MirixClient(
            base_url=args.mirix_url,
            user_id=args.user_id,
            timeout=args.mirix_timeout,
        )
        if not await mirix.health():
            logger.error("MIRIX server not reachable at %s. "
                         "Start it with: python scripts/start_server.py --port 8531",
                         args.mirix_url)
            return 2
        evolver = MirixSkillEvolver(mirix_client=mirix)
        skill_mgr = MirixSkillManager(mirix_client=mirix)

    # OpenAI client for the agent loop
    openai_client = None
    chat_model = os.environ.get("EVAL_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    if not args.dry_run:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_API_BASE", OPENROUTER_BASE_URL),
        )

    summary = {"run_id": run_id, "days": [], "started_at": dt.datetime.now().isoformat()}

    for day in days:
        logger.info("=== %s ===", day)
        q = _load_questions(day)
        rounds = q.get("rounds", [])
        if args.max_rounds:
            rounds = rounds[: args.max_rounds]
        round_results: list[RoundResult] = []

        for r in rounds:
            round_id = r["id"]
            round_type = r["type"]
            question = r["question"]
            eval_block = r.get("eval", {})

            if args.dry_run:
                # No LLM, no MIRIX — produce a deterministic stub for plumbing tests
                from evals.metaclaw.format_adapter import RoundResult as RR
                rr = RR(
                    round_id=round_id, round_type=round_type, question=question,
                    final_answer="(dry-run)", reward=0.0, eval_outcome="fail",
                    feedback=_expected_feedback(r, "fail"), error="dry_run",
                )
                round_results.append(rr)
                logger.info("[%s/%s] dry-run", day, round_id)
                continue

            skills = skill_mgr.retrieve(question, top_k=args.top_k)
            logger.info("[%s/%s] retrieved %d skills", day, round_id, len(skills))
            cfg = RunnerConfig(
                chat_model=chat_model, workspace=workspace,
                max_turns=args.max_turns, wallclock_cap_s=args.wallclock_cap,
            )
            rr = run_round(
                openai_client=openai_client, cfg=cfg,
                round_id=round_id, round_type=round_type,
                question=question, eval_block=eval_block, skills=skills,
            )
            rr.feedback = _expected_feedback(r, rr.eval_outcome)
            round_results.append(rr)
            logger.info("[%s/%s] outcome=%s reward=%s",
                        day, round_id, rr.eval_outcome, rr.reward)

        # Day-end evolve
        evolve_status = "skipped"
        diff_summary = {"created": [], "edited": [], "deleted": []}
        if not args.dry_run and not args.no_evolve:
            try:
                metaclaw_skills = await evolver.evolve(round_results, current_skills={})
                evolve_status = "ok"
                diff_summary = {
                    "produced_skills": [s["name"] for s in metaclaw_skills],
                }
                logger.info("[%s] evolve produced %d skills: %s",
                            day, len(metaclaw_skills),
                            [s["name"] for s in metaclaw_skills])
            except Exception as e:
                evolve_status = f"failed:{type(e).__name__}:{e}"
                logger.warning("[%s] evolve failed: %s", day, e)

        # Per-day metrics
        n = len(round_results)
        passed = sum(1 for r in round_results if r.reward >= 1.0)
        per_round = [
            {"id": r.round_id, "type": r.round_type, "outcome": r.eval_outcome,
             "reward": r.reward, "error": r.error}
            for r in round_results
        ]
        day_metrics = {
            "day": day, "n_rounds": n, "n_passed": passed,
            "pass_rate": (passed / n) if n else 0.0,
            "per_round": per_round,
            "evolve_status": evolve_status, "evolve_diff": diff_summary,
        }
        (run_dir / f"{day}_metrics.json").write_text(
            json.dumps(day_metrics, indent=2, default=str)
        )
        summary["days"].append(day_metrics)

    summary["finished_at"] = dt.datetime.now().isoformat()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_summary_md(run_dir, summary)
    logger.info("Done. Summary at %s/summary.md", run_dir)

    if mirix is not None:
        await mirix.aclose()
    return 0


def _write_summary_md(run_dir: Path, summary: dict) -> None:
    lines = ["# MIRIX × MetaClaw Eval Summary", "",
             f"Run id: `{summary['run_id']}`",
             f"Started: {summary['started_at']}",
             f"Finished: {summary['finished_at']}", "",
             "## Per-day pass rate", "",
             "| Day | Rounds | Passed | Pass rate | Evolve | Skills produced |",
             "|---|---|---|---|---|---|"]
    for d in summary["days"]:
        produced = d["evolve_diff"].get("produced_skills", [])
        lines.append(
            f"| {d['day']} | {d['n_rounds']} | {d['n_passed']} | "
            f"{d['pass_rate']:.2f} | {d['evolve_status']} | "
            f"{', '.join(produced) or '—'} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", nargs="+", choices=["day01", "day02", "day03"],
                   help="Subset of days to run (default: all three).")
    p.add_argument("--max-rounds", type=int, default=0,
                   help="Cap rounds per day (0 = no cap; useful for smoke).")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--wallclock-cap", type=float, default=300.0)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--user-id", type=str, default="eval-metaclaw-3day")
    p.add_argument("--mirix-url", type=str, default="http://127.0.0.1:8531")
    p.add_argument("--mirix-timeout", type=float, default=600.0)
    p.add_argument("--no-evolve", action="store_true",
                   help="Skip day-end evolve calls (debug).")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip all LLM and MIRIX calls; emit stub metrics for plumbing tests.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:]) if argv is None else argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9.2: Smoke the driver in dry-run mode (no LLM, no MIRIX)**

```bash
python -m evals.metaclaw.run_3day_eval --days day01 --max-rounds 1 --dry-run
```

Expected: exits 0, prints `Run <id> — output dir ...`, then
`Workspace prepared at ...`, then a `dry-run` log line per round, then
`Done.` Verify outputs:

```bash
ls evals/metaclaw/reports/                              # at least one run dir
cat evals/metaclaw/reports/*/day01_metrics.json | head -30   # JSON with day=day01, error=dry_run
cat evals/metaclaw/reports/*/summary.md                # markdown table
```

- [ ] **Step 9.3: Add `evals/metaclaw/reports/` to `.gitignore`**

Append to `.gitignore`:

```
# Eval outputs
evals/metaclaw/reports/
```

- [ ] **Step 9.4: Commit driver**

```bash
git add evals/metaclaw/run_3day_eval.py .gitignore
git commit -m "[VEPEAGE-000] eval: add run_3day_eval driver (dry-run smoked)"
```

---

## Task 10: Configure MIRIX to use OpenRouter and start the server

This is **environmental setup** — no code change committed, but the runtime
state has to be right before Task 11 succeeds.

- [ ] **Step 10.1: Confirm OpenRouter key is exported**

User must `export OPENAI_API_KEY=<openrouter-key>`. Verify:

```bash
test -n "$OPENAI_API_KEY" && echo "OK" || echo "MISSING"
echo "$OPENAI_API_KEY" | head -c 6
```

Expected: `OK` and a prefix like `sk-or-`. If `MISSING`, ask the user to
paste the key into the current shell or uncomment the relevant line in
`~/.zshrc`.

- [ ] **Step 10.2: Export the rest of the eval env**

```bash
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export EVAL_CHAT_MODEL="openai/gpt-5.2"
export EVAL_EMBED_MODEL="google/gemini-embedding-001"
export EVAL_EMBED_DIM="1536"
```

- [ ] **Step 10.3: Verify MIRIX's default LLMConfig in DB matches OpenRouter**

```bash
docker exec e2e-postgres psql -U mirix -d mirix -c "
  SELECT model, model_endpoint_type, model_endpoint
  FROM agents
  ORDER BY created_at DESC LIMIT 5;
" 2>&1 | head -20
```

If existing agents point at `https://api.openai.com/v1` (default) we need
to either (a) rebuild them before evolve uses them, or (b) accept that the
*first* invocation creates fresh sub-agents using whatever the
`scripts/start_server.py` initialization does. The path of least
resistance is **(c) override at process-start time**: when starting the
MIRIX server, set the env vars above so its sub-agent factories pick up
the OpenRouter values for newly-created agents. The dedicated eval user
(`eval-metaclaw-3day`) does not yet exist, so its meta-agent will be
created on the first `/v1/skills/evolve` call with these env values
in scope.

- [ ] **Step 10.4: Start the MIRIX server in a separate terminal**

In a **new terminal** (so the env above is inherited), from MIRIX repo root:

```bash
python scripts/start_server.py --port 8531
```

Expected: log lines including `Uvicorn running on http://0.0.0.0:8531`.
Leave this terminal running for all subsequent tasks.

- [ ] **Step 10.5: Health-check from the original terminal**

```bash
curl -fsS http://127.0.0.1:8531/ | head -c 200
echo
```

Expected: a non-empty 200 response. If the path `/` does not respond,
try `/healthz` or `/docs`:

```bash
curl -fsS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8531/docs
```

`200` indicates the server is up.

If `MirixClient.health()` (which probes `/`) gets a non-200 from this
deployment, edit `evals/metaclaw/mirix_client.py` to probe `/docs` or
`/healthz` instead and re-run its tests. (One-line change; commit
separately if so.)

- [ ] **Step 10.6: Ensure the `eval-metaclaw-3day` user exists**

`POST /v1/skills/evolve` returns 404 if the `user_id` does not exist.
The REST endpoint `POST /users/create_or_get` requires client auth, which
is more setup than we need here. Insert the user directly via psql:

```bash
docker exec e2e-postgres psql -U mirix -d mirix -c "
  INSERT INTO users (id, name, organization_id, timezone, status,
                     created_at, updated_at, is_deleted)
  SELECT 'eval-metaclaw-3day',
         'eval-metaclaw-3day',
         (SELECT id FROM organizations LIMIT 1),
         'UTC',
         'active',
         NOW(), NOW(), false
  WHERE NOT EXISTS (SELECT 1 FROM users WHERE id = 'eval-metaclaw-3day');
"
```

Verify:

```bash
docker exec e2e-postgres psql -U mirix -d mirix -c \
  "SELECT id, name, organization_id, status FROM users WHERE id='eval-metaclaw-3day';"
```

Expected: one row.

If the `users` table schema differs (column missing, FK constraint), inspect
it first:

```bash
docker exec e2e-postgres psql -U mirix -d mirix -c "\d users"
```

Adjust the INSERT accordingly. The intent is: a single row in `users`
keyed `eval-metaclaw-3day`, joined to whatever organization the rest of
the MIRIX deployment uses.

---

## Task 11: Smoke run — one real round end-to-end

- [ ] **Step 11.1: Run a single real round on day01**

From the eval-shell terminal:

```bash
python -m evals.metaclaw.run_3day_eval --days day01 --max-rounds 1
```

Expected:
- Driver logs `retrieved 0 skills` (cold start — MIRIX has nothing yet).
- Agent loop runs (you'll see one or more LLM round-trips in the MIRIX
  server log too if any tool calls hit MIRIX).
- One round completes with `outcome=pass` or `outcome=fail` (cold-start
  fail on ISO 8601 is the expected case for the first run).
- Day-end evolve fires; log line `evolve produced N skills: [...]` (likely
  one skill named something like `iso8601-with-cst-offset` — the exact
  name is up to MIRIX's ProceduralMemoryAgent).
- A `reports/<run-id>/` directory exists with `day01_metrics.json` and
  `summary.md`.

- [ ] **Step 11.2: Inspect the produced skill in MIRIX**

```bash
curl -s "http://127.0.0.1:8531/v1/skills?user_id=eval-metaclaw-3day&limit=10" \
  | python -m json.tool | head -40
```

Expected: at least one skill object with `name`, `description`,
`instructions` related to ISO 8601 / datetime formatting.

- [ ] **Step 11.3: Decide whether to proceed to full e2e**

If Step 11.1 emitted any of:
- `MIRIX server not reachable` → fix Task 10 first.
- `evolve_status: failed:...` → check MIRIX server log; if it's about
  embedding dimension or auth, adjust env (`EVAL_EMBED_DIM`, key) and
  retry the smoke.
- `error="api_timeout"` or `error="turn_limit"` on the round → this is
  acceptable for cold start; proceed.

If the smoke produced at least one metric file and the evolve call
succeeded, proceed to Task 12.

---

## Task 12: Full 3-day e2e run

- [ ] **Step 12.1: Reset MIRIX skills for the eval user (clean slate)**

```bash
curl -s "http://127.0.0.1:8531/v1/skills?user_id=eval-metaclaw-3day&limit=1000" \
  | python -c "import json,sys,subprocess; \
    skills=json.load(sys.stdin).get('skills',[]); \
    [subprocess.run(['curl','-s','-X','DELETE', \
      f'http://127.0.0.1:8531/v1/skills/{s[\"id\"]}?user_id=eval-metaclaw-3day']) \
     for s in skills]; print(f'deleted {len(skills)}')"
```

Expected: `deleted N` (N matches what the smoke created).

- [ ] **Step 12.2: Run all three days**

```bash
python -m evals.metaclaw.run_3day_eval 2>&1 | tee /tmp/mirix-metaclaw-eval-$(date +%s).log
```

Expected duration: roughly 5–25 minutes depending on OpenRouter latency
and how many tool turns the agent uses per round. Estimated LLM cost is
modest (a few dollars at GPT-5.2 OpenRouter prices).

While it runs, watch for:
- `[dayNN/rXX] outcome=pass` lines — the prevalence of `pass` after day01
  is the proxy for "MIRIX learned the preference."
- Per-day `evolve produced N skills` log lines.

- [ ] **Step 12.3: Verify the report**

```bash
RUN=$(ls -t evals/metaclaw/reports | head -1)
cat evals/metaclaw/reports/$RUN/summary.md
```

Expected: a 3-row markdown table like

```
| Day | Rounds | Passed | Pass rate | Evolve | Skills produced |
|---|---|---|---|---|---|
| day01 | 5 | 0 | 0.00 | ok | iso8601-with-cst-offset |
| day02 | 7 | 5 | 0.71 | ok | iso8601-with-cst-offset (edited) |
| day03 | 6 | 5 | 0.83 | ok | (no new skills) |
```

(Numbers are illustrative — the real shape is what we want to read off
to compare against the paper's day-by-day P1 accuracy curve.)

- [ ] **Step 12.4: Inspect the evolution trajectory in MIRIX**

```bash
curl -s "http://127.0.0.1:8531/v1/skills?user_id=eval-metaclaw-3day&limit=50" \
  | python -m json.tool > evals/metaclaw/reports/$RUN/final_skill_bank.json
cat evals/metaclaw/reports/$RUN/final_skill_bank.json | head -60
```

Expected: a small skill bank (1–5 skills) all about datetime / ISO 8601 /
output format. Each skill has a `version` field — repeated edits across
days bump the patch number.

- [ ] **Step 12.5: Commit the final summary as a record**

The `reports/` dir is gitignored (Step 9.3), but we want this run's
`summary.md` and `summary.json` checked in for traceability:

```bash
mkdir -p docs/superpowers/eval-results
cp evals/metaclaw/reports/$RUN/summary.md \
   docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.md
cp evals/metaclaw/reports/$RUN/summary.json \
   docs/superpowers/eval-results/2026-05-08-metaclaw-3day-summary.json
git add -f docs/superpowers/eval-results/
git commit -m "[VEPEAGE-000] eval: record 3-day MIRIX × MetaClaw summary"
```

---

## Task 13: Final sweep — full test suite + cleanup

- [ ] **Step 13.1: Run the full eval test suite**

```bash
pytest evals/metaclaw/tests -v
```

Expected: all green. Total ~17 tests across 4 test modules.

- [ ] **Step 13.2: Check no MIRIX core tests regressed**

```bash
pytest tests/test_skill_cli_tools.py tests/test_skill_orm.py tests/test_skill_schema.py -v
```

Expected: same green status as on baseline `feat/skill-evolve` HEAD before
this plan was applied.

- [ ] **Step 13.3: Stop the MIRIX server**

In the terminal running `start_server.py`, hit Ctrl-C.

- [ ] **Step 13.4: Final commit (if anything dangling)**

```bash
git status
```

If clean, no further commit. If anything dangling from earlier tasks
slipped (e.g. README polish, log file cleanup), squash a final commit:

```bash
git add <files>
git commit -m "[VEPEAGE-000] eval: tidy"
```

---

## Spec coverage checklist

| Spec section | Implemented in |
|---|---|
| §2 D1 (day01..day03) | Tasks 9.1, 12.2 |
| §2 D2 (single arm) | Whole plan; no baseline arm tasks present (intentional) |
| §2 D3 (reuse metaclaw classes, not its rollout) | Tasks 1, 6, 7, 8 |
| §2 D4 (OpenRouter via OpenAI SDK) | Task 4, Task 9 driver `OpenAI(...base_url...)` |
| §2 D5 (chat model `openai/gpt-5.2`) | Task 4 + 10.2 |
| §2 D6 (embedding `google/gemini-embedding-001`) | Task 4 + 10.2 |
| §2 D7 (MIRIX REST: evolve + retrieval) | Tasks 5, 6, 7 |
| §2 D8 (one message per round) | Task 6 (MirixSkillEvolver) |
| §2 D9 (subclass SkillEvolver) | Task 6 |
| §2 D10 (subclass SkillManager) | Task 7 |
| §2 D11 (no preload built-ins) | Task 12.1 (clean slate before full run) |
| §2 D12 (workspace once, carried) | Task 9 `_prepare_workspace` (called once per run) |
| §2 D13 (MIRIX as single source of truth) | No SKILL.md writeback in any task |
| §3 dataset analysis | Used as expected outcome in Task 12.3 |
| §4 architecture | Tasks 5–9 implement; Tasks 10–12 run |
| §5 repo layout | Tasks 1–9 create exactly that layout |
| §6 component contracts | Implemented file-by-file (Tasks 3–9) |
| §7 cross-day data flow | Task 9 driver loop |
| §8 error handling | Tasks 9 (driver), 8 (turn cap), 10 (health-check) |
| §9 testing | Tasks 3, 5, 6, 7, 8, 13 |
| §10 risks | Risk on `metaclaw rollout` → resolved by writing thin loop ourselves (Task 8); risk on embedding dim → handled by `EVAL_EMBED_DIM` env (Task 4) |
| §11 out of scope | No tasks for baseline arm, days 04+, RL, OpenClaw GUI, P2 |
