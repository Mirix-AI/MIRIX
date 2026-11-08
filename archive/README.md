# Archive — superseded code

Modules retired from the live tree because nothing in the current design
(graph_version v7.x / v8: hypergraph ingest hooks, V7Retriever with rerank,
gated auto_dream) reaches them. Preserved here — and in full per-version
history at the `graph_revision_pre_squash` tag — rather than deleted.

## legacy_graph/

The pre-v7 graph generations and rejected v7 extraction variants:

- `episodic_graph_manager.py`, `semantic_graph_manager.py`, `lightrag_merger.py`
  — the v5 dual-graph ingest builders (Episode/EpisodicEntity/Concept labels).
- `episodic_graph_retriever.py`, `semantic_graph_retriever.py`,
  `lightrag_keyword_extractor.py`, `_graph_retriever_base.py` — the v5
  retrieval pipeline (keyword-LLM call + per-graph retrievers + budget split).
- `graph_memory_manager_v6.py`, `graph_retriever_v6.py` — the v6 lean entity
  index.
- `proposition_extractor.py` — v7.3 proposition ingest (rejected: collapsed the
  edge structure).
- `gliner_extractor.py` — v7.4 GLiNER extraction (superseded by direction D:
  a span tagger cannot abstract concepts).

The dispatcher and the episodic/semantic ingest hooks now warn-and-skip for
graph versions outside the v7 family instead of running these.

Also removed from live files (recoverable from git history, documented in
`docs/graph_memory_v7/development_history.md`): the v7.7 PPR and v7.2 coverage
retrieval branches and the v7.9 role-split rendering in `graph_retriever_v7.py`,
and the rejected answerer gates (v7.11 consolidate tool, persona store,
graph-search merge) in `evals/task_agent.py`. A second audited sweep also
removed: the never-measured MIRIX_GRAPH_ROUTED_SEARCH gate and its helper in
rest_api.py, the v5/v6 Neo4j DDL and neo4j_healthcheck, the LightRAG relation
output surface (the live v7/v8 path reads entities only), four orphaned
lightrag prompt symbols, token_tracker.set_phase, the vestigial dispatcher
budget parameters, and EpisodicEventUpdate.source_refs.

## evals/

Eval-side scripts for rejected or superseded experiments: `build_persona.py`
(persona store, QA-negative), `refine_hypergraph.py` (one-off corpus cleanup,
superseded by ingest-time prevention + maintenance passes),
`proposition_ingest.py` (v7.3 driver), `run_v4_sample0.sh`,
`visualize_v6_graph.py`, `draw_hypergraph.py` (viz one-off). (`memory_snapshot.py` was initially archived here too,
then restored — the MAB runner scripts call it for post-run snapshots; the
"unreferenced" scan had only covered *.py, not *.sh.)

## rejected/

`semantic_consolidator.py` — the auto_dream "v2" flat-store stage. It clustered
near-duplicate semantic memory rows and had an LLM rewrite each cluster into one
additive replacement row (originals kept, provenance edges written back). It was
removed because it consolidated the WRONG THING: the requirement was that
consolidation happen on graph nodes and leave the flat store alone, and even an
additive row changes flat retrieval — it is indexed, competes for top-k and
shifts answers. Measured before removal on LongMemEval-S: 36/60 vs 34/60
no-dream (inside the noise band) for ~6.3h of LLM time per run, with merged rows
averaging 2.3k chars against 640 for a raw row and compounding as products were
re-merged. Published work agrees the trade is bad at this scale (Kang et al.
2026 measure merging as net-harmful once the raw rows already fit the context;
Zhang et al. 2026 measure a 42-point decay from consolidating consolidations).
Node merging in `graph_reconsolidator` — value union plus edge rewiring, no text
generation — is the mechanism that was actually wanted, and it costs about a
minute.
