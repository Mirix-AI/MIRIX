# v6 Graph Memory Revision Notes

Lean entity-index graph memory for MIRIX. v6 keeps Neo4j as a lightweight
connector between extracted entities and PostgreSQL flat memory rows, rather
than storing full memory content or dense relation descriptions in the graph.

## Related Docs

- `docs/graph_memory_v2.md` - original PostgreSQL graph-memory layer.
- `docs/graph_memory_v4/README.md` - LightRAG-style dual graph patch summary.
- `docs/graph_memory_v4/v4_graph_memory.md` - full v4 source/diff archive.
- `docs/graph_memory_v7/README.md` - minimal semantic+episodic linkage graph
  revision that keeps details in flat memory.

This file records the v6 revision and the LongMemEval-S smoke run. It is not a
replacement for the v4 docs; it is meant to make v6 comparable against earlier
graph designs.

## Design

v6 treats the graph as an inverted index:

1. Store entity nodes in Neo4j.
2. Link those entities to episodic and semantic flat-memory rows.
3. Retrieve by entity-name vector search plus one-hop co-occurrence expansion.
4. Fetch full episodic/semantic details from PostgreSQL.
5. Format the graph-linked flat details into the QA prompt.

The important invariant is that Neo4j should improve linking and recall, while
PostgreSQL remains the source of detailed memory content.

## Current Graph Shape

The LongMemEval-S v6 graph currently uses this shape:

```text
(:V6Entity)-[:APPEARS_IN]->(:V6MemoryRef:V6EpisodeRef {id: "episodic:<pg_id>"})
(:V6Entity)-[:DESCRIBED_BY]->(:V6MemoryRef:V6ConceptRef {id: "semantic:<pg_id>"})
```

`V6_COOCCUR` is part of the lean-index design, but this LongMem-S snapshot has
0 co-occurrence edges. The generated visualizations therefore show the actual
entity-to-memory-ref bipartite graph.

Older v6 code also expected this property-array shape:

```text
(:V6Entity {episodic_ids: [<pg_id>], semantic_ids: [<pg_id>]})
```

The retriever now accepts both shapes so older runs and current `V6MemoryRef`
runs can be compared without rebuilding the graph.

## Visualizations

Generated from the current `longmem_s_0` v6 Neo4j graph:

- [Interactive HTML index](visualizations/index.html)
- [Overview: top-degree entities](visualizations/overview_top_entities.svg)
- [Aquarium / fish detail](visualizations/topic_aquarium.svg)
- [Career / campaign work detail](visualizations/topic_career.svg)
- [Fitness / health devices detail](visualizations/topic_fitness.svg)
- [Travel / places detail](visualizations/topic_travel.svg)

The full graph has 7,281 nodes and 14,125 edges, so these are sampled views.
Entity nodes are on the left; `V6MemoryRef` nodes are on the right. Yellow
memory nodes are episodic refs, blue memory nodes are semantic refs. Blue edges
mean `APPEARS_IN`; purple edges mean `DESCRIBED_BY`.

Regenerate with:

```bash
set -a; source /Users/weichiehhuang/MIRIX_eval/.env; set +a
/Users/weichiehhuang/MIRIX_eval/.venv/bin/python scripts/visualize_v6_graph.py \
  --user-id longmem_s_0 \
  --out-dir docs/graph_memory_v6/visualizations
```

## LongMemEval-S Run

Run date: 2026-06-18  
Dataset: LongMemEval-S, `longmem_s_0`  
Scope: 1 conversation, 534 chunks, 60 QA  
Mode: graph enabled, `MIRIX_GRAPH_VERSION=v6`  
Result folder:
`evals/results/longmem/longmemS_gmemS_v6_60qa_judge_clean/`

Stored memory size after ingest:

| Store | Count | Stored chars |
|---|---:|---:|
| Episodic flat memory | 702 rows | 417,922 |
| Semantic flat memory | 566 rows | 372,347 |
| Neo4j graph | 7,281 nodes / 14,125 edges | 742,271 |

QA runtime for the 60 questions:

| Phase | Total sec | Avg sec |
|---|---:|---:|
| Retrieval / prompt wrap | 157.51 | 2.63 |
| Answer generation | 240.79 | 4.01 |
| Total QA wall time | 398.30 | 6.64 |

## Judge Result

Judge model: `gpt-4o-mini` via `evals/organize_results.py` / `evals/llm_judge.py`.

| Category | n | Correct | Accuracy |
|---|---:|---:|---:|
| knowledge-update | 9 | 7 | 77.78% |
| multi-session | 15 | 9 | 60.00% |
| single-session-assistant | 6 | 6 | 100.00% |
| single-session-preference | 6 | 4 | 66.67% |
| single-session-user | 9 | 7 | 77.78% |
| temporal-reasoning | 15 | 10 | 66.67% |
| **Overall** | **60** | **43** | **71.67%** |

Metrics file:
`evals/results/longmem/longmemS_gmemS_v6_60qa_judge_clean/metrics.json`

## Cross-Run Comparison

Timing columns use eval JSON timings:

- `Ingest min` = `timings.add_chunk` total.
- `QA min` = `timings.wrap_user_prompt + timings.answer`.
- `Retrieve min` = `timings.wrap_user_prompt`.
- `Answer min` = `timings.answer`.

Node/edge columns come from `memory_stats.graph` when the run recorded it.
Rows with different chunk counts are useful directional references, but the
closest apples-to-apples LongMem comparison is `Small-chunk graph` vs `v6
gmemS` because both use 534 chunks and the same 60 QA set.

### LongMemEval-S `longmem_s_0`, 60 QA

| Setting | Chunks | Correct | Accuracy | Nodes | Edges | Graph chars | Ingest min | QA min | Retrieve min | Answer min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No graph full | 391 | 39/60 | 65.00% | 0 | 0 | 0 | 135.3 | 4.72 | 1.23 | 3.49 |
| Graph full | 391 | 36/60 | 60.00% | 8,590 | 21,635 | 2,155,921 | 485.3 | 6.95 | 3.19 | 3.76 |
| Graph v2 session chunk | 111 | 38/60 | 63.33% | 11,218 | 28,646 | 2,643,620 | 344.2 | 8.36 | 3.79 | 4.56 |
| Graph v2 recency 0.3 | 111 | 38/60 | 63.33% | 11,218 | 28,646 | 2,643,620 | 344.2 | 9.46 | 4.28 | 5.18 |
| Small-chunk graph | 534 | 44/60 | 73.33% | 9,130 | 23,658 | 2,361,276 | 444.8 | 8.15 | 3.62 | 4.53 |
| v6 gmemS | 534 | 43/60 | 71.67% | 7,281 | 14,125 | 742,271 | resume | 6.64 | 2.63 | 4.01 |

Against the closest 534-chunk baseline (`Small-chunk graph`), v6 is slightly
lower on accuracy (-1/60, -1.67 pp) but much smaller and faster at QA:

| Delta: v6 vs small-chunk graph | Value |
|---|---:|
| Accuracy | -1.67 pp |
| Nodes | -20.3% |
| Edges | -40.3% |
| Graph stored chars | -68.6% |
| QA time | -18.5% |
| Retrieval / prompt-wrap time | -27.4% |
| Answer time | -11.4% |

Size deltas against older graph settings:

| Delta: v6 vs | Nodes | Edges | Graph chars |
|---|---:|---:|---:|
| Graph full | -15.2% | -34.7% | -65.6% |
| Graph v2 session chunk | -35.1% | -50.7% | -71.9% |
| Small-chunk graph | -20.3% | -40.3% | -68.6% |

### LoCoMo `conv-26`, Historical Runs

These results are the available single-conversation LoCoMo metrics from
`evals/results/locomo`. They have accuracy and timing, but those older result
JSON files did not record graph node/edge counts, and the live Neo4j database
has since been reused for other runs.

| Run | Correct | Accuracy | Nodes | Edges | Ingest min | QA min | Retrieve min | Answer min | Avg answer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0201c no graph | 133/152 | 87.50% | -- | -- | 9.9 | 7.17 | 0.25 | 6.92 | 2.70s |
| 0201c_graph_v5 | 137/152 | 90.13% | -- | -- | 42.2 | 12.75 | 0.06 | 12.69 | 4.94s |
| 0201c_graph_v6 | 134/152 | 88.16% | -- | -- | 36.2 | 10.55 | 0.10 | 10.44 | 4.07s |
| 0201c_graph_v7_run1 | 112/152 | 73.68% | -- | -- | 16.4 | 11.78 | 3.91 | 7.86 | 3.06s |
| 0201c_v5_promptfix_graph | 127/152 | 83.55% | -- | -- | 32.5 | 17.21 | 7.76 | 9.45 | 3.68s |
| 0201c_v5_triggerfix | 130/152 | 85.53% | -- | -- | 8.7 | 12.69 | 3.34 | 9.35 | 3.64s |
| early graph | 129/152 | 84.87% | -- | -- | 29.5 | 18.83 | 9.72 | 9.11 | 3.55s |
| graph_v5_premerge_backup | 128/152 | 84.21% | -- | -- | 30.3 | 17.80 | 7.59 | 10.20 | 3.98s |
| premerge no graph backup | 122/152 | 80.26% | -- | -- | 8.8 | 16.01 | 3.61 | 12.40 | 4.83s |

### Published v2 LoCoMo Full-Benchmark Result

`docs/graph_memory_v2.md` records a broader 10-conversation LoCoMo benchmark
(1540 judged questions). That run is not directly comparable to the single
`conv-26` and LongMem smoke runs above, but it provides useful historical
context:

| Setting | LLM Judge | Wall time |
|---|---:|---:|
| No graph | 54.29% | 2.7 h |
| v2 graph | 57.34% | 3.4 h |
| Delta | +3.05 pp | +42 min (+26%) |

## Diagnosis From This Run

The first QA attempt looked graph-only because the retriever returned only
matched entity names:

```text
## Memory Index (v6)
Matched entities: ...
```

That happened for two separate compatibility reasons:

1. The existing Neo4j graph stored links through `V6MemoryRef` nodes, while the
   checked-out v6 retriever only read `V6Entity.episodic_ids` and
   `V6Entity.semantic_ids`.
2. Flat fallback search hit PostgreSQL schema drift: the current ORM selected
   `source_refs` / `prior_values`, but the existing `mirix_v6_longmems`
   database did not have those columns yet.

Fixes applied:

- `mirix/services/graph_retriever_v6.py` now collects backrefs from both
  `V6MemoryRef` edges and legacy entity arrays, strips `episodic:` /
  `semantic:` prefixes, then fetches the matching PG rows.
- `evals/longmem_eval.py` now measures PostgreSQL flat memory using
  `MIRIX_PG_DB`, `MIRIX_PG_HOST`, `MIRIX_PG_PORT`, `MIRIX_PG_USER`, and
  `MIRIX_PG_PASSWORD` instead of hard-coding database `mirix`.
- `mirix_v6_longmems` was migrated with empty JSONB defaults for
  `episodic_memory.source_refs`, `semantic_memory.source_refs`, and
  `semantic_memory.prior_values`.

After the retriever fix, QA prompts included both entity matches and full
episodic / semantic flat details.

## Error Pattern

The remaining 17 judged wrong answers are mostly not linkage failures. They
cluster around:

- multi-session counting and aggregation, such as fish count, fitness classes,
  health devices, jewelry count, and current-role duration;
- temporal comparison or ordering, such as seed-start order, phone case versus
  charger, Valentine's Day airline, Yosemite trip length, and January sports
  event order;
- single conflicting fact retrieval, such as handbag amount, farmers market
  earning, and internet-plan speed.

This suggests the v6 graph-to-flat path is usable, but the next revision should
focus on ranking enough linked details for aggregation and adding stronger
temporal/negative-evidence handling.

## Notes

- The original result folder also contains
  `longmem_s_0.graph_only_bad24.json`, a partial 24-question run from before
  the retriever compatibility fix. It is intentionally excluded from the clean
  judge folder.
- `metrics_contaminated_with_bad24.json` in the resume folder should not be
  used for reporting; it included the partial graph-only run.
