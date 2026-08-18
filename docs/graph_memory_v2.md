# MIRIX v2: Graph Memory Layer

**Author**: JasonH  
**Date**: 2026-04-03  
**Status**: Implemented & Evaluated

---

## Overview

Graph Memory adds a temporal knowledge graph on top of MIRIX's existing flat episodic and semantic memory. When enabled, it extracts entities and relations from conversations and stores them as a graph (nodes + edges) in PostgreSQL, enabling multi-hop traversal and temporal reasoning during retrieval.

**Key result**: +3.05% LLM Judge accuracy on LoCoMo benchmark (1540 questions, 10 conversations) with +26% time overhead.

---

## Architecture

```
Conversation Input
       │
       ▼
 ┌─────────────────────────┐
 │  Meta Memory Manager     │  (unchanged)
 │  → Episodic Agent        │
 │  → Semantic Agent        │
 │  → Core/Procedural/etc   │
 └─────┬──────────┬─────────┘
       │          │
       ▼          ▼
 ┌──────────┐ ┌──────────┐
 │ Flat     │ │ Graph    │  ← NEW (toggle: MIRIX_ENABLE_GRAPH_MEMORY)
 │ Memory   │ │ Memory   │
 │ (v1)     │ │ (v2)     │
 └──────────┘ └──────────┘
       │          │
       ▼          ▼
    PostgreSQL + pgvector
```

When `MIRIX_ENABLE_GRAPH_MEMORY=true`:
- **Write path**: After each episodic/semantic memory insert, additionally extract entities + relations into the graph
- **Read path**: During retrieval, supplement flat memory results with graph-traversed facts + episodes

When disabled (default): zero changes to behavior, zero overhead.

---

## Database Schema

4 new tables (auto-created by SQLAlchemy):

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `entity_nodes` | Named entities (people, places, concepts) | name, entity_type, embedding, summary |
| `entity_edges` | Semantic facts between entities (bi-temporal) | src_id, dst_id, rel_type, fact_text, valid_at, invalid_at, expired_at |
| `episode_nodes` | Timestamped events | summary, details, event_time, embedding |
| `involves_edges` | Cross-links episode ↔ entity | episode_id, entity_id, role |

---

## Write Path (W1–W5)

Triggered at the end of `insert_event()` and `insert_semantic_item()`.

| Step | What | LLM Calls |
|------|------|-----------|
| W1 | Create episode_node (summary + embedding) | 0 |
| W2 | Extract entities + relations from text (structured JSON output) | **1** |
| W3 | Entity dedup (case-insensitive name match by user_id) | 0 |
| W4 | Edge insert + conflict detection (expire old edge if same src+rel_type) | 0 |
| W5 | Create involves_edges (episode ↔ entity links) | 0 |

**Total: 1 LLM call per memory insert** (vs. Graphiti's 6–10).

### W2 Extraction Prompt

Single structured-output call extracts:
```json
{
  "entities": [{"name": "Caroline", "type": "PERSON"}],
  "relations": [{"src": "Caroline", "rel_type": "WORKS_AT", "dst": "Meta", "fact_text": "...", "valid_at": "..."}],
  "episode_entities": ["Caroline", "Meta"]
}
```

### W4 Conflict Detection

If a new edge has the same `(src_id, rel_type)` as an existing active edge with different `fact_text`:
- Old edge gets `expired_at = now()`
- New edge is inserted
- History is preserved (not deleted)

---

## Read Path (R1–R4)

Triggered in `retrieve_memories_by_keywords()` in rest_api.py.

### R1: Seed Discovery (union of 4 signals)

| Signal | What | Purpose |
|--------|------|---------|
| R1a | BM25 keyword search on entity names (top 3 query words) | Name matching |
| R1b | Embedding similarity search on entity_nodes | Semantic matching |
| R1c | Most recent 5 entities | Recency coverage |
| R1d | BM25 search on entity_edges.fact_text | Find edges directly (critical for temporal) |

All results are unioned — no mutual exclusion.

### R2: 2-Hop Graph Expansion

```
Seed entities → Hop 1 neighbors (via entity_edges) → All edges + episodes touching expanded set
```

SQL uses explicit hop tracking for scoring.

### R3: Scoring & Pruning

```python
score = 0.5 * cosine_sim(query_emb, candidate_emb)  # semantic relevance
      + 0.3 * exp(-0.693 * age_days / 30)             # recency (valid_at, 30-day half-life)
      + 0.2 * (1 / (1 + hop_distance))                # proximity (0=seed, 1=neighbor)
```

Top 15 edges + top 5 episodes selected.

### R4: Context Formatting

```
## Relevant Facts (from knowledge graph)
- Caroline attended an LGBTQ support group (on/since 07 May 2023)
- Melanie painted a lake sunrise (on/since 15 March 2022)

## Recent Related Events (from knowledge graph)
- [08 May 2023] Caroline shared her experience at an LGBTQ support group...
```

Full dates (`%d %B %Y`) for temporal reasoning.

---

## Code Changes

### Modified Files (6 files, +59 lines, 0 deletions)

| File | Change |
|------|--------|
| `mirix/settings.py` | Add `enable_graph_memory: bool = Field(False, env="MIRIX_ENABLE_GRAPH_MEMORY")` |
| `mirix/orm/__init__.py` | Register 4 new ORM models |
| `mirix/server/server.py` | Import + init `GraphMemoryManager` |
| `mirix/server/rest_api.py` | Add graph retrieval in `retrieve_memories_by_keywords()` |
| `mirix/services/episodic_memory_manager.py` | Add graph write hook in `insert_event()` |
| `mirix/services/semantic_memory_manager.py` | Add graph write hook in `insert_semantic_item()` |

### New Files (2 files)

| File | Purpose |
|------|---------|
| `mirix/orm/graph_memory.py` | ORM models: EntityNode, EntityEdge, EpisodeNode, InvolvesEdge |
| `mirix/services/graph_memory_manager.py` | Write path (W1–W5) + Read path (R1–R4) |

### Design Principles

- All graph code is behind `if settings.enable_graph_memory` — default off
- Graph hooks are `try/except` non-fatal — graph failure doesn't affect v1 memory
- No original MIRIX logic is altered — only additions at the end of existing functions
- No new external dependencies — uses existing PostgreSQL (pgvector) + OpenAI API

---

## Evaluation: LoCoMo Benchmark

### Setup

- **Dataset**: LoCoMo 10 conversations, 1540 questions (excl. adversarial)
- **Model**: gpt-4.1-mini (memory extraction + QA + judge)
- **Embedding**: text-embedding-3-small (1536 dim)
- **Protocol**: Per the original MIRIX eval — each conversation gets fresh DB, all sessions ingested, then all questions answered
- **Judge**: Binary CORRECT/WRONG (original MIRIX eval metric)

### Results

| Metric | No Graph | With Graph | Delta |
|--------|---------|------------|-------|
| **LLM Judge** | 0.5429 | **0.5734** | **+0.0305 (+5.6%)** |
| Token F1 | 0.3255 | 0.3285 | +0.0030 |
| BLEU-1 | 0.2698 | 0.2730 | +0.0031 |

### Per Category (LLM Judge)

| Category | n | No Graph | Graph | Delta |
|----------|---|---------|-------|-------|
| **Open Domain** | 96 | 0.3646 | **0.4271** | **+0.0625** |
| **Single Hop** | 282 | 0.5461 | **0.5957** | **+0.0496** |
| **Temporal** | 841 | 0.5541 | **0.5874** | **+0.0333** |
| Multi-Hop | 321 | 0.5639 | 0.5607 | -0.0031 |

### Timing

| | No Graph | With Graph | Overhead |
|--|---------|------------|----------|
| Wall time | 2.7 hours | 3.4 hours | +42 min (+26%) |

### Analysis

1. **Open Domain (+6.25%)**: Largest gain — graph entity/relation context enriches open-ended answers
2. **Single Hop (+4.96%)**: Direct fact retrieval from entity_edges
3. **Temporal (+3.33%)**: R1d edge BM25 search + full-date formatting helps time reasoning
4. **Multi-Hop (-0.31%)**: Essentially flat — confirmed not a regression (was -9.6% on single sample due to noise)
5. **Cost**: +1 LLM call per memory insert, ~26% more time overall

---

## Usage

### Enable Graph Memory

```bash
# Server-side toggle
MIRIX_ENABLE_GRAPH_MEMORY=true python scripts/start_server.py --port 8531

# Or in .env
MIRIX_ENABLE_GRAPH_MEMORY=true
```

### Run Evaluation

```bash
# Without graph (baseline)
python tests/run_locomo_all.py

# With graph
python tests/run_locomo_all.py --graph

# Single sample
python tests/test_locomo_quick.py --sample 0

# Both modes + comparison
bash public_evaluations/run.sh
```

### Check Graph Data

```sql
SELECT 'entity_nodes' as tbl, count(*) FROM entity_nodes
UNION ALL SELECT 'entity_edges', count(*) FROM entity_edges
UNION ALL SELECT 'episode_nodes', count(*) FROM episode_nodes
UNION ALL SELECT 'involves_edges', count(*) FROM involves_edges;
```

---

## Future Work

1. **Multi-hop scoring**: Use Personalized PageRank instead of simple 2-hop BFS for better multi-hop retrieval
2. **Entity summary updates**: LLM call to update entity summaries on dedup merge (currently only creates)
3. **Conflict detection with LLM**: Add LLM confirmation for ambiguous edge conflicts (currently auto-expires)
4. **Embedding-based edge search in R1**: Add `embedding <=> query` search on entity_edges alongside BM25
5. **Adaptive top_k**: Dynamically adjust number of graph facts based on query complexity
