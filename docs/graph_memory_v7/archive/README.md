# Archive — superseded design-era documents

Process records from earlier phases of the graph-memory work, preserved verbatim
for the paper trail. The designs they describe have been superseded or absorbed:

- `10chunk_ingest_analysis.md` — early controlled 10-chunk ingest study (v7 era,
  pre-hypergraph). Its broken-ingest pitfall findings were folded into the eval
  harness; the graph-shape analysis predates the v7.10 hypergraph.
- `extractor_direction_D.md` — the research memo that selected LLM triple
  extraction ("direction D") over GLiNER/LightRAG. Direction D shipped as v7.6
  and remains the current extractor; the memo's speed claims were later
  corrected (2.6×, not 34×).
- `v74_v78_summary.md` — consolidated summary of the extractor line v7.4–v7.8
  plus the first QA-failure study. Superseded by the version ledger and the
  later per-question forensics.
- `hypergraph_and_consolidate.md` — v7.10 hypergraph build log and the v7.11
  `consolidate` answerer tool (measured negative, rejected; the counting gap it
  targeted is now addressed by consolidation-side design instead).

For the current state of the system, read `../version_ledger.md`,
`../development_history.md`, and the two results records
(`../extraction_recovery.md`, `../locomo_dream_ablation.md`).
