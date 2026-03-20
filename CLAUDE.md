# MIRIX - Claude Code Instructions

## What This Project Does
MIRIX is an async-native multi-agent personal assistant with a six-component memory system
(Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault). It captures memories
from conversations and screen activity, storing them in PostgreSQL with vector search via pgvector.

## Key Architecture
- **Entry point**: `mirix/server/rest_api.py` (FastAPI, port 8531)
- **Orchestration**: `mirix/agent/meta_agent.py` → 6 sub-agents
- **Data flow**: ORM (`mirix/orm/`) → Schemas (`mirix/schemas/`) → Managers (`mirix/services/`) → API (`mirix/server/`)
- **Queue**: Kafka-backed (`mirix/queue/`) for async memory extraction
- **All I/O is async** — never introduce sync blocking calls (see `docs/Mirix_async_native_changes.md`)

## Local Development Setup

### Prerequisites
- Docker + Docker Compose
- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, or Google Gemini)

### Start Infrastructure
```bash
cp docker/env.example .env   # then add your API keys to .env
docker-compose up -d         # starts PostgreSQL (5432), Redis (6379), API (8531), Dashboard (5173)
docker-compose ps            # verify all services healthy
```

### Run API Locally (without Docker)
```bash
pip install -e .
export GEMINI_API_KEY=your-key   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
python scripts/start_server.py --port 8531
```

### Access Points
- Dashboard: http://localhost:5173
- API Swagger: http://localhost:8531/docs
- API ReDoc: http://localhost:8531/redoc

## Running Tests

The preferred way to run tests is via the dockerized test script, which handles infrastructure automatically:

```bash
# Full suite with verbose output (preferred)
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO

# Pass any pytest args after the flags
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO -k test_message_handling
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO -m "not integration"
```

**Required env var for tests**: `GEMINI_API_KEY`

### Running without Docker (manual infra)
```bash
# Fast unit tests — no running server needed (~20s)
pytest tests/test_memory_server.py -v

# All tests except integration
pytest -m "not integration" -v

# Integration tests — requires server on port 8899
python scripts/start_server.py --port 8899          # Terminal 1
pytest tests/test_memory_integration.py -v -m integration -s   # Terminal 2
```

## Common Dev Tasks

### Add a new API endpoint
1. Add Pydantic request/response schemas to `mirix/schemas/`
2. Add business logic method to the relevant manager in `mirix/services/`
3. Add the route to `mirix/server/rest_api.py`
4. Add the corresponding method to `mirix/client/remote_client.py`

### Add a new memory type
1. Create ORM model in `mirix/orm/`
2. Create Pydantic schemas in `mirix/schemas/`
3. Create manager in `mirix/services/`
4. Create sub-agent in `mirix/agent/`
5. Register in `mirix/agent/meta_agent.py`

### Format & lint
```bash
# Preferred (poetry)
poetry run black . && poetry run isort .

# Alternatively via make
make format   # ruff import sort + format
make lint     # ruff check + pyright
make check    # format + lint + test
```

## Async Rules (Critical)
- **All new manager methods must be `async def`**
- **Never use `asyncio.run()` inside the server** — the event loop is already running
- Only 5 intentional sync touch-points exist (LangFuse, Gmail OAuth, SQLAlchemy DDL at startup, cleanup job entry, pure CPU helpers) — do not add more
- Use `asyncio.to_thread()` to wrap any unavoidably sync third-party calls

## Commit Convention
Prefix commits with the Jira ticket: `[VEPEAGE-NNN] Description`

## Do Not
- Confuse queue messages (transient) with database messages (persistent)
- Call `step_manager.get_step()` — steps are write-only audit logs
- Skip `create_or_get_user()` — always ensure users exist first
- Commit secrets or API keys
- Create README or doc files unless explicitly requested
