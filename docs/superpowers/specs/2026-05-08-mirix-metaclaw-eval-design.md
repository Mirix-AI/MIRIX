# MIRIX × MetaClaw Evolution-Bench Eval Harness — Design Spec

## 1. Overview

Replace MetaClaw's built-in `SkillEvolver` and `SkillManager` with MIRIX's
procedural memory subsystem, then run MetaClaw's own day-by-day Evolution-Bench
(days 01–03) to measure whether MIRIX's evolve + retrieval improves an agent's
ability to learn user preferences across days.

The experiment scope is intentionally narrow: **single-arm**, MIRIX-only,
3 days of data; the paper's published table is the comparison point. This is
an integration and measurement harness, not a new training stack.

## 2. Locked Decisions

| # | Decision | Choice |
|---|---|---|
| D1 | Dataset | `MetaClaw/benchmark/data/metaclaw-bench/eval/day01..day03/questions.json` (3-day P1 arc — ISO 8601 datetime preference) |
| D2 | Experiment design | Single arm (MIRIX); the paper's published per-day accuracy table is the baseline reference |
| D3 | Agent runtime | `pip install -e MetaClaw` repo, reuse its `rollout` + `agent_loop` (paper bench path; not OpenClaw GUI) |
| D4 | LLM router | OpenRouter, OpenAI Python SDK with `base_url=https://openrouter.ai/api/v1` |
| D5 | Chat model | `openai/gpt-5.2` (used by both the bench agent and MIRIX internal sub-agents) |
| D6 | Embedding model | `google/gemini-embedding-001` via OpenRouter (same key, same base URL) |
| D7 | MIRIX integration | REST: `POST /v1/skills/evolve` and `GET /v1/skills?query=...&search_method=bm25` |
| D8 | Evolve message granularity | One message per round (success and failure both sent) |
| D9 | Evolver replacement | Subclass `metaclaw.skill_evolver.SkillEvolver`; `evolve()` calls MIRIX REST |
| D10 | Retrieval replacement | Subclass `metaclaw.skill_manager.SkillManager`; `retrieve()` calls MIRIX REST |
| D11 | Built-in skills | Not preloaded (cold start, paper-aligned). Driver flag `--preload-builtins` provided as escape hatch |
| D12 | Workspace | Copied once from `metaclaw-bench/workspaces/shared/` at start; carried across days (paper convention) |
| D13 | MIRIX storage | Single source of truth in MIRIX Postgres; no `~/.metaclaw/skills/` SKILL.md files |

## 3. Dataset Analysis

All three days target the **same P1 preference**: ISO 8601 datetime fields
with a `+08:00` timezone offset (e.g. `2026-03-16T09:30:00+08:00`).

| Day | Theme | P1 role | Rounds |
|---|---|---|---|
| day01 | Sprint 7 standup notes | First introduction | 5 |
| day02 | Sprint 7 milestones / progress report | Consolidation | 7 |
| day03 | API access log analysis / event triage | Cross-domain transfer | 6 |

Every `file_check` round's eval command is `python scripts/check_iso8601.py
<file> <fields>`; every `multi_choice` round is graded by exact match against
`eval.answer`. The bench is therefore a clean, quantitative test of how well
the evolver can lift the day-N+1 first-attempt accuracy after seeing day-N
feedback.

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│              evals/metaclaw/run_3day_eval.py  (driver)               │
│                day01 → day02 → day03 in sequence                     │
└──────────────────────────────────────────────────────────────────────┘
        │ (monkey-patches metaclaw two classes at import time)
        ▼
┌──────────────────────────────┐    ┌──────────────────────────────────┐
│  metaclaw rollout +          │    │ MirixSkillEvolver                │
│  agent_loop (paper code,     │    │   evolve()                       │
│  unmodified)                 │    │     → POST /v1/skills/evolve     │
│   - questions.json driver    │    │ MirixSkillManager                │
│   - bench eval scoring       │    │   retrieve()                     │
│   - Qwen3-native tool calls  │    │     → GET  /v1/skills?query=     │
└──────────────────────────────┘    └──────────────────────────────────┘
        │                                        │
        ▼                                        ▼
┌──────────────────────────────┐    ┌──────────────────────────────────┐
│  OpenRouter                  │    │ MIRIX REST API (:8531)           │
│   chat:  openai/gpt-5.2      │    │   chat: openai/gpt-5.2 (same)    │
│   embed: gemini-embedding-001│    │   embed: gemini-embedding-001    │
└──────────────────────────────┘    │   PostgreSQL + pgvector          │
                                    └──────────────────────────────────┘
```

Three runtime processes: (1) MIRIX API on `:8531`, (2) the eval driver,
(3) OpenRouter (remote). PostgreSQL already runs locally via the existing
`docker-compose` stack.

## 5. Repository Layout

```
MIRIX/
├── evals/metaclaw/                       (new)
│   ├── run_3day_eval.py                  driver entry point
│   ├── mirix_skill_evolver.py            subclass of metaclaw.SkillEvolver
│   ├── mirix_skill_manager.py            subclass of metaclaw.SkillManager
│   ├── mirix_client.py                   thin httpx wrapper around MIRIX REST
│   ├── format_adapter.py                 schema bridge MIRIX ↔ metaclaw
│   ├── round_runner.py                   single-round agent loop + eval scoring
│   ├── reports/<run-id>/                 metrics + summary outputs (gitignored)
│   ├── tests/                            unit tests
│   └── README.md                         how to run
├── third_party/MetaClaw/                 (gitignored) git clone of aiming-lab/MetaClaw
└── ... (existing MIRIX source unchanged)
```

Bootstrap commands (once, at the start of writing-plans / implementation):

```
git clone https://github.com/aiming-lab/MetaClaw third_party/MetaClaw
cd third_party/MetaClaw && pip install -e ".[evolve]"   # skips RL extras
```

## 6. Component Contracts

### 6.1 `MirixClient`
```python
class MirixClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8531",
                 user_id: str): ...
    async def evolve(self, messages: list[str]) -> dict:
        """POST /v1/skills/evolve → {created: [...], edited: [...], deleted: [...]}"""
    async def search_skills(self, query: str, limit: int = 6) -> list[dict]:
        """GET /v1/skills?query=...&search_method=bm25 → list of skill dicts"""
    async def health(self) -> bool:
        """Health probe for driver startup."""
```

### 6.2 `format_adapter`

```python
def mirix_to_metaclaw(skill: dict) -> dict:
    return {
        "name":        skill["name"],
        "description": skill["description"],
        "content":     skill["instructions"],
        "category":    skill.get("entry_type", "general"),
    }

def round_to_message(round_result: RoundResult) -> str:
    """Serialize one round into a human-readable message containing
       round_id, question, agent's final answer, eval outcome (pass/fail),
       and the bench's feedback (correct/incorrect text)."""
```

### 6.3 `MirixSkillEvolver` (subclass)

```python
from metaclaw.skill_evolver import SkillEvolver

class MirixSkillEvolver(SkillEvolver):
    def __init__(self, mirix_client: MirixClient, ...): ...
    def should_evolve(self, batch, threshold=0.0) -> bool:
        return True  # driver triggers explicitly at day-end
    async def evolve(self, samples, current_skills) -> list[dict]:
        messages = [round_to_message(s) for s in samples]
        diff     = await self.mirix.evolve(messages)
        return [mirix_to_metaclaw(s) for s in diff["created"] + diff["edited"]]
```

### 6.4 `MirixSkillManager` (subclass)

```python
from metaclaw.skill_manager import SkillManager

class MirixSkillManager(SkillManager):
    def __init__(self, mirix_client: MirixClient, ...):
        # Skip the parent's SKILL.md scan; skills always come from MIRIX.
        self.mirix = mirix_client
    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        skills = run_sync(self.mirix.search_skills(query, limit=top_k))
        return [mirix_to_metaclaw(s) for s in skills]
```

### 6.5 `run_3day_eval.py` (driver)

Sequence:

1. Health-check MIRIX API on `:8531`; abort with start command if down.
2. Verify MIRIX `LLMConfig` is set to OpenRouter chat + Gemini embedding;
   fail-fast if mismatched. Concretely the driver expects, at startup:
   - `OPENAI_API_KEY` set to the OpenRouter key (MIRIX's openai client
     accepts arbitrary `base_url`, see `mirix/llm_api/openai_client.py:82`).
   - `OPENAI_API_BASE=https://openrouter.ai/api/v1`.
   - MIRIX `LLMConfig.model_endpoint_type="openai"`,
     `model_endpoint="https://openrouter.ai/api/v1"`,
     `model="openai/gpt-5.2"` for chat;
     `embedding_endpoint_type="openai"`,
     `embedding_endpoint="https://openrouter.ai/api/v1"`,
     `embedding_model="google/gemini-embedding-001"` for embeddings.
3. Resolve / create a dedicated MIRIX `user_id` (e.g. `eval-metaclaw-3day`).
4. Optional: if `--preload-builtins`, ingest the 36 built-in SKILL.md
   directories into MIRIX as the starting skill bank.
5. Copy `metaclaw-bench/workspaces/shared/` into a per-run scratch
   directory `runs/<run-id>/workspace/`. This workspace persists across
   the three days.
6. For each day in `["day01", "day02", "day03"]`:
   a. Load `eval/<day>/questions.json`.
   b. For each round:
      - Retrieve top-k skills via `MirixSkillManager.retrieve(question)`.
      - Compose the agent prompt (system + skills + question), drive the
        agent loop in the bench workspace, capture transcript.
      - Score: run `eval.command` for `file_check`, exact-match for
        `multi_choice`. Reward = 1.0 on pass, 0.0 on fail.
   c. Day-end: build `messages = [round_to_message(r) for r in rounds]`;
      call `MirixSkillEvolver.evolve(messages, current_skills)`.
   d. Write `reports/<run-id>/<day>_metrics.json`:
      `{day, n_rounds, n_passed, pass_rate, per_round: [...], evolve_diff}`.
7. Aggregate `summary.md` with per-day pass rates, evolution trajectory
   (skill names created/edited per day), and pointers to the metrics files.

## 7. Cross-Day Data Flow

```
day01 start
  workspace ← cp -r metaclaw-bench/workspaces/shared/  (clean baseline)
  for round r in day01.rounds:
     skills ← GET MIRIX /v1/skills?query=<r.question>&search_method=bm25
     prompt = system + skills + r.question
     loop:
        LLM call (openai/gpt-5.2 via OpenRouter) → tool calls
        execute bash/read/write tools in workspace
        until done or turn cap
     reward = eval(r) ∈ {0.0, 1.0}
     append round_result
  day-end:
     POST MIRIX /v1/skills/evolve {messages: [round_to_message(r) for r in rounds]}
     diff stored in MIRIX postgres + copied to day01_metrics.json

day02 start
  workspace ← (carried from day01, including artefacts day01 produced)
  for round r in day02.rounds:
     skills ← MIRIX retrieve  (now includes day01-learned ISO 8601 skill)
     ...
  day-end: evolve again

day03 start
  same; expect transfer of the ISO 8601 skill into the API-log domain.
```

## 8. Error Handling

| Scenario | Behaviour |
|---|---|
| MIRIX server not up | Driver health-check fails, exit non-zero with start command hint |
| OpenRouter timeout / 429 | OpenAI SDK auto-retry (3× exponential); after exhaustion mark round `reward=0, error="api_timeout"` and continue |
| `eval.command` script error (vs. assertion failure) | Record `reward=0, error="eval_script_error"`, distinguish from "agent answered wrong" in the summary |
| MIRIX evolve call fails | Skip that day's evolve, set `evolve_status="failed"`, continue to next day with previous skills |
| Agent turn-loop runaway | Per-round caps: `max_turns=20`, wallclock cap 5 min; over-cap → `reward=0, error="turn_limit"` |
| Embedding dimension mismatch | Set `output_dimensionality` on Gemini embedding to MIRIX's `MAX_EMBEDDING_DIM`; assert at startup |

## 9. Testing Strategy

**Unit (`evals/metaclaw/tests/`):**

1. `test_format_adapter.py` — round-trip `MIRIX skill ↔ metaclaw skill` and
   `RoundResult → message` serialization.
2. `test_mirix_client.py` — httpx mock; verify request bodies for `evolve`
   and `search_skills` match the MIRIX REST contract; verify response parsing.
3. `test_round_runner.py` — mock LLM and a mock workspace; verify a
   `file_check` round computes reward correctly and a `multi_choice` round
   parses `\bbox{A,E}` answers.

**Integration:**

1. **Smoke**: `python -m evals.metaclaw.run_3day_eval --days day01
   --max-rounds 1` — single round end-to-end against a live MIRIX server
   to validate the full chain (retrieve → agent → score → evolve).
2. **Real e2e**: full `day01..day03` run; this is the deliverable the
   user requested.

## 10. Key Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| metaclaw `[evolve]` extras pull in heavy / Mac-incompatible deps | Medium | Install as `pip install -e ".[evolve]"` only; document any required `--no-deps` overrides in the eval README |
| `metaclaw.ConversationSample` schema differs from our `RoundResult` | Medium | `format_adapter` handles both directions; covered by unit tests |
| MIRIX evolve runtime exceeds wallclock budget (ProceduralMemoryAgent multi-step CLI workflow + GPT-5.2 latency) | Medium | Server timeout 300 s, driver wait 600 s; evolve failure does not block subsequent days |
| GPT-5.2 model id on OpenRouter not exactly `openai/gpt-5.2` | Low | Driver checks `/api/v1/models` at startup and surfaces the resolved id |
| metaclaw rollout in `skills_only` mode hard-couples to Tinker (RL-only path) | Medium | Verified at writing-plans time by reading source; if so, replicate a thin ~150-line agent loop matching metaclaw's Qwen3-native tool-call protocol — semantically equivalent for our purposes |
| pgvector dimension incompatibility with `gemini-embedding-001` | Low | Use Gemini's `output_dimensionality` parameter to truncate to MIRIX's column dim |

## 11. Out of Scope

- Running a baseline arm with the original `SkillEvolver` (the paper's
  published per-day accuracy table is the comparison reference).
- Days 04–30 of the bench.
- The `arc=B` / P2 (file_naming) preference — only `arc=A` / P1 (output_format)
  is in this 3-day slice.
- RL training, Tinker / MinT / Weaver backends, PRM scoring, OPD distillation.
- The OpenClaw / CoPaw / IronClaw GUI clients (not on the paper's bench
  evaluation path).
- Public benchmark publication, leaderboard submission, or paper claims.
- Any modification to MIRIX's existing skill-evolve internals (Phase 2 already
  shipped on `feat/skill-evolve` is consumed as-is).
