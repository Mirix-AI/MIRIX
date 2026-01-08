# Mirix Docker Setup

## Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **PostgreSQL** | `ankane/pgvector:v0.5.1` | 5432 | Vector database with pgvector extension |
| **Redis Stack** | `redis/redis-stack-server:latest` | 6379 | High-performance caching & vector search |
| **Mirix API** | `mirix-api:latest` (built locally) | 8531 | REST API backend server |
| **Dashboard** | `mirix-dashboard:latest` (built locally) | 5173 | React web UI |

## Quick Start

```bash
# 1. Copy environment file and add your API keys
cp docker/env.example .env
# Edit .env and set at least OPENAI_API_KEY

# 2. Start all services (builds images on first run)
docker-compose up -d

# 3. Verify services
docker-compose ps

# 4. Access the dashboard
# Open http://localhost:5173 in your browser
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:5173 | Web UI for managing Mirix |
| **API Swagger** | http://localhost:8531/docs | API documentation |
| **API ReDoc** | http://localhost:8531/redoc | Alternative API docs |

## Environment Variables

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required for embeddings) |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRIX_PG_USER` | `mirix` | PostgreSQL username |
| `MIRIX_PG_PASSWORD` | `mirix` | PostgreSQL password |
| `MIRIX_PG_DB` | `mirix` | PostgreSQL database name |
| `MIRIX_REDIS_ENABLED` | `true` | Enable Redis caching |

### Optional LLM Providers

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `AZURE_API_KEY` | Azure OpenAI API key |
| `OLLAMA_BASE_URL` | Ollama server URL (e.g., `http://host.docker.internal:11434`) |

## Data Persistence

Data is stored in `.persist/` directory:
- `.persist/pgdata/` - PostgreSQL data
- `.persist/redis-data/` - Redis data
- `.persist/mirix-data/` - Mirix application data

## Common Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f mirix_api
docker-compose logs -f dashboard

# Rebuild after code changes
docker-compose build
docker-compose up -d

# Rebuild specific service
docker-compose build dashboard
docker-compose up -d dashboard

# Stop services
docker-compose down

# Reset everything (WARNING: deletes all data)
docker-compose down -v
rm -rf .persist/
docker-compose up -d
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │────▶│   Mirix API     │
│   (port 5173)   │     │   (port 8531)   │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌───────────────┐         ┌───────────────┐
           │  PostgreSQL   │         │    Redis      │
           │  (port 5432)  │         │  (port 6379)  │
           └───────────────┘         └───────────────┘
```

## Troubleshooting

```bash
# Check service health
docker-compose ps

# View all logs
docker-compose logs

# Connect to PostgreSQL
docker exec -it mirix_pgvector psql -U mirix -d mirix

# Connect to Redis
docker exec -it mirix_redis redis-cli ping

# Check API health
curl http://localhost:8531/health

# Restart a service
docker-compose restart mirix_api
docker-compose restart dashboard
```
