# v7 Graph Memory: Minimal Semantic+Episodic Linkage

v7 is the "honmei" graph revision: details stay in flat memory, and graph
nodes/edges must earn their place by creating a useful retrieval or reasoning
path.

It is intentionally side-by-side with v6. Use `MIRIX_GRAPH_VERSION=v7` to run
it; v5/v6 labels and previous results remain comparable.

## Related Docs

- `docs/graph_memory_v6/README.md` - v6 design, LongMem-S run, judge result,
  and v6 graph visualization links.
- `docs/graph_memory_v6/visualizations/index.html` - sampled v6 graph views.

## Principle

The graph should not carry the memory content. PostgreSQL flat memory remains
the source of detail:

- `episodic_memory`: events, timestamps, numbers, concrete evidence.
- `semantic_memory`: stable facts, preferences, identities, durable concepts.
- Neo4j graph: sparse anchors and necessary links between semantic and
  episodic memory refs.

The core rule is:

> If removing a node or edge does not remove an important retrieval path, it
> probably should not be in the graph.

## Schema

```text
(:V7Anchor)
  - id
  - user_id
  - organization_id
  - name
  - name_lower
  - anchor_type
  - name_embedding

(:V7MemoryRef:V7EpisodeRef)
  - id = "episodic:<pg_id>"
  - memory_id = <episodic_memory.id>
  - memory_type = "episodic"
  - source_key
  - timestamp

(:V7MemoryRef:V7ConceptRef)
  - id = "semantic:<pg_id>"
  - memory_id = <semantic_memory.id>
  - memory_type = "semantic"
  - source_key

(:V7Anchor)-[:V7_APPEARS_IN]->(:V7EpisodeRef)
(:V7Anchor)-[:V7_DESCRIBED_BY]->(:V7ConceptRef)
(:V7ConceptRef)-[:V7_SUPPORTED_BY]->(:V7EpisodeRef)
(:V7EpisodeRef)-[:V7_NEXT_MEMORY]->(:V7EpisodeRef)
```

`V7MemoryRef` nodes store only ids, timestamps, provenance keys, and short
debug previews. Full summaries/details are fetched from PostgreSQL during
retrieval.

## What Changed From v6

| Area | v6 | v7 |
|---|---|---|
| Entity admission | Keeps most extracted entities | Filters generic noun phrases and keeps specific anchors only |
| Entity/entity edges | Optional co-occurrence | Removed |
| Memory refs | v6 supported both arrays and `V6MemoryRef` compatibility | Explicit `V7MemoryRef` nodes |
| Semantic+episodic bridge | Present in some snapshots as `SUPPORTED_BY` | First-class `V7_SUPPORTED_BY` via shared `source_meta` |
| Details in graph | v6 snapshots duplicated summaries in graph refs | v7 stores only preview/title metadata; details stay in PG |
| Retrieval | Anchor/entity search -> PG fetch | Anchor search -> semantic/episodic refs -> support expansion -> PG fetch |

## Anchor Admission

v7 still reuses the existing LightRAG extraction call, but applies a gate before
writing graph nodes. It favors:

- people, locations, organizations, named events;
- owned or recurring objects;
- named content, products, venues, apps, classes, pets, trips;
- specific concepts with proper names, numbers, or multi-word identity.

It rejects generic anchors like:

- `Tips`
- `Advice`
- `Flexibility`
- `Information`
- `Recommendations`
- `Methods`
- `Options`

This is deliberately conservative. The goal is to reduce graph size and make
each remaining anchor feel necessary.

## Write Path

Both episodic and semantic memory managers route into
`V7GraphManager.process_memory()` when:

```bash
MIRIX_ENABLE_GRAPH_MEMORY=true
MIRIX_GRAPH_VERSION=v7
```

For each PG memory row:

1. Create/merge one `V7MemoryRef`.
2. Extract candidate entities from the memory text.
3. Filter candidates through the anchor-admission gate.
4. Merge selected `V7Anchor` nodes.
5. Link anchors to the memory ref.
6. If `source_meta` exists, link semantic refs to episodic refs from the same
   source chunk with `V7_SUPPORTED_BY`.
7. For episodic refs, add `V7_NEXT_MEMORY` to preserve chronology.

## Read Path

`V7Retriever` runs when `MIRIX_GRAPH_VERSION=v7`:

1. Embed query.
2. Vector search `V7Anchor.name_embedding`.
3. Collect direct episodic refs and semantic refs.
4. Expand semantic refs to supporting episodic refs.
5. Expand episodic refs to nearby temporal refs.
6. Fetch full details from `episodic_memory` and `semantic_memory` in PG.
7. Format a compact context:

```text
## Memory Linkage Graph (v7)
Matched anchors: ...

### Semantic memories (PG flat)
...

### Episodic memories (PG flat evidence)
...
```

## Files

- `mirix/services/graph_memory_manager_v7.py`
- `mirix/services/graph_retriever_v7.py`
- `mirix/database/neo4j_client.py` adds v7 constraints + anchor vector index.
- `mirix/services/episodic_memory_manager.py` routes episodic inserts.
- `mirix/services/semantic_memory_manager.py` routes semantic inserts.
- `mirix/services/graph_retriever_dispatcher.py` routes retrieval.

## Run Command

Example server env:

```bash
export MIRIX_ENABLE_GRAPH_MEMORY=true
export MIRIX_GRAPH_VERSION=v7
export MIRIX_PG_DB=mirix_v7_longmems
export MIRIX_NEO4J_URI=bolt://localhost:7687
export MIRIX_NEO4J_USER=neo4j
export MIRIX_NEO4J_PASSWORD=mirix_neo4j_dev
```

Use a clean PG DB and clean Neo4j label set when benchmarking so v7 counts are
not mixed with v5/v6 runs.

## Smoke Validation

Run date: 2026-06-18  
Scope: local smoke only. This validates v7 wiring, schema, and graph writes; it
is not a LongMem or LoCoMo accuracy run.

Checks completed:

- Import and anchor gate: `V7GraphManager` and `V7Retriever` import cleanly.
- Anchor admission kept specific anchors:
  `Natural Park of Moncayo Mountain`, `American Airlines`,
  `20-Gallon Community Aquarium`, `Luna`.
- Anchor admission rejected generic anchors:
  `Tips`, `Flexibility`, `Social Media`.
- Neo4j schema bootstrap connected to `bolt://localhost:7687`.
- v7 indexes observed:
  `v7_anchor_name_emb`, `v7_memory_ref_user_source`.
- v7 write path used a mocked extractor and mocked 1536-dim embeddings, then
  wrote one episodic memory ref and one semantic memory ref with shared
  `source_meta.chunk_id = v7-smoke-chunk-001`.

Smoke graph counts before cleanup:

| Item | Count |
|---|---:|
| `V7Anchor` | 3 |
| `V7EpisodeRef` | 1 |
| `V7ConceptRef` | 1 |
| `V7_APPEARS_IN` | 2 |
| `V7_DESCRIBED_BY` | 2 |
| `V7_SUPPORTED_BY` | 1 |

Smoke cleanup deleted all temporary nodes for user id `__v7_smoke_user__`;
post-cleanup counts were all 0.

## Expected Evaluation Questions

v7 should be compared against the closest v6/small-chunk LongMem setup on:

- accuracy;
- node count;
- edge count;
- graph stored chars;
- ingest time;
- retrieval / prompt-wrap time;
- answer time.

The target is not necessarily higher raw accuracy on the first run. The target
is a smaller graph whose remaining nodes/edges are easier to justify, while
keeping accuracy close enough that better ranking/support expansion can recover
the last few missed questions.
