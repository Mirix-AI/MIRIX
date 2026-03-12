# Mirix Tests

## Setup

```bash
# Install dependencies
pip install -e .

# Set API key
export GEMINI_API_KEY=your_api_key_here
# Or on Windows:
# set GEMINI_API_KEY=your_api_key_here
```

## Test Files Overview

| File | Tests | Type | Speed | Description |
|------|-------|------|-------|-------------|
| `test_memory_server.py` | 18 | Unit | Fast (~20s) | Direct `AsyncServer()` calls, no network |
| `test_memory_integration.py` | 4 | Integration | Slow (~45s) | REST API via real server + client |

## Run Tests

```bash
# Run all tests (server + integration)
pytest -v

# Server-side tests only (fast, no real server needed)
pytest tests/test_memory_server.py -v

# Integration tests only (requires manually started server - see below)
pytest tests/test_memory_integration.py -v -m integration -s

# Skip integration tests (runs server tests only)
pytest -m "not integration" -v
```

## Test Coverage

### Server Tests (`test_memory_server.py`)
Comprehensive coverage of all 5 memory types with all search methods:
- ✅ **Episodic Memory**: Insert events, search by summary/details (bm25, embedding)
- ✅ **Procedural Memory**: Insert procedures, search by summary/steps (bm25, embedding)
- ✅ **Resource Memory**: Insert resources, search by summary (bm25, embedding) / content (bm25 only)
- ✅ **Knowledge Vault**: Insert knowledge, search by caption (bm25, embedding) / secret_value (bm25)
- ✅ **Semantic Memory**: Insert items, search by name/summary/details (bm25, embedding)
- ✅ **Cross-memory search**: Search across all memory types

### Integration Tests (`test_memory_integration.py`)
Core API operations via client-server:
- ✅ `client.add()`: Add memories via conversation
- ✅ `client.retrieve_with_conversation()`: Retrieve with context
- ✅ `client.retrieve_with_topic()`: Retrieve by topic
- ✅ `client.search()`: Search memories (bm25, embedding)

## Prerequisites

**API Key Required**: Set `GEMINI_API_KEY` environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
# Or on Windows:
set GEMINI_API_KEY=your_api_key_here
```

**Automatic Initialization**: Both test files will automatically:
- Create `demo-user` in `demo-org` organization
- Initialize meta agent and all sub-agents (episodic, procedural, resource, knowledge vault, semantic)
- No manual setup needed!

## Running Integration Tests

Integration tests require a **manually started server** on port 8899:

```bash
# Terminal 1: Start server
python scripts/start_server.py --port 8899

# Terminal 2: Run integration tests (will auto-initialize on first run)
pytest tests/test_memory_integration.py -v -m integration -s
```

**Note**: Server tests (`test_memory_server.py`) don't need a running server.

## Common Options

```bash
# Show print statements
pytest -v -s

# Run specific test
pytest tests/test_memory_server.py::TestDirectEpisodicMemory::test_insert_event -v

# With coverage
pytest --cov=mirix --cov-report=html

# Debug on failure
pytest --pdb
```
