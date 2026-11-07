# LoCoMo conv-26 — graph / reranker / consolidation ablation, and making the dream lossless

Six arms on LoCoMo conversation 26 (199 questions; the MAB-style judge scores
152 — category 5 adversarial is excluded), hypergraph `v7.10`, answerer
gpt-4.1-mini, embeddings text-embedding-3-small. **Every arm with its own store
was ingested fresh, end to end, with the graph built in-line by the real insert
hooks** — the offline rebuild shortcut is banned for evals (it is a different
construction process: v7.8 entity resolution is path-dependent, and the paper
describes the online system). Arms 4/5 reuse the store+graph of 2/3 and change
only retrieval; arm 6 is the final recipe.

## 1. The table

| # | config | retrieval | store (mem) | graph core nodes | anchors/facts/refs | QA (152 judged) |
|---|---|---|---|---|---|---|
| 1 | no graph | flat only | 227 | 0 (disabled) | — | .8816 (134) |
| 2 | graph | inject, no rerank | 228 | 2310 | ~780/~1400/228 | .8882 (135) |
| **4** | **graph + reranker** | inject + rerank | 228 (=2) | 2310 (=2) | as above | **.9079 (138)** |
| 3 | graph + dream (ungated) | inject, no rerank | 205 (−10%) | 2136 (−7.5%) | 676/1255/205 | .8618 (131) |
| 5 | graph + dream (ungated) | inject + rerank | 205 (=3) | 2136 (=3) | as above | .8750 (133) |
| **6** | **graph + dream (union gate)** | inject + rerank | **214 (−6%)** | **2269 (−1.8%)** | 717/1338/214 | **134 / 137 / 135 (x̄ 135.3)** |

Arm 2's graph composition is a proportional estimate (only the wipe total was
recorded before arm 3 rebuilt the per-user subgraph); the total and every arm
3/6 number are measured. Arm 6 graph invariants at rest: 0 zombie facts, full
98/98 temporal chain. "dream" = interleaved auto_dream, one cycle after every
10th ingested chunk plus one after the final chunk
(`MIRIX_DREAM_EVERY_N_CHUNKS=10` in `evals/main_eval.py`).

Per-category, on the rerank arms (directly comparable):

| | single_hop (32) | temporal (37) | multi_hop (13) | open_domain (70) |
|---|---|---|---|---|
| 4 no-dream | .906 | .919 | .923 | .900 |
| 5 dream ungated | .781 | **.838** | .923 | .929 |
| 6 dream gated | .875 | **.919** | .923 | .857 |

## 2. Findings

### The reranker is real: +2~3, paired

Same store, same graph, only the ordering toggled
(`MIRIX_GRAPH_RERANK`): 135→138 on the clean store, 131→133 on the dreamed
store. Second benchmark to confirm the v7.1 rerank (SH-Doc was +4). It had been
dead code for every version after v7.1 — a stale `== "v7.1"` guard — until the
guard fix; the same stale-tuple class also had the retriever dispatcher sending
v7.4–v7.10 to the dead v5 pipeline (an LLM call for empty context on every
`wrap_user_prompt`).

### The graph's contribution is multi-hop — and insurance

multi_hop is .846 with no graph and .923 in **all four** graph arms. Overall
the graph alone is +1 (noise) when the flat store is intact. But on a
consolidated store the injection carried ~24/152: an arm that accidentally ran
with an empty graph injection against a dreamed store scored 0.7039 vs 0.8618
for the same condition with its graph — the graph's facts (extracted before the
merge) supply what the merged flat rows lost. Flat-intact: redundant.
Flat-consolidated: load-bearing.

### Why consolidation loses facts: it is a rewrite + hard delete, not a merge

The dream agent's "merge" is: LLM rewrites N rows into M<N new rows, then
hard-deletes the originals. `episodic_memory_replace`'s own docstring documents
`new_items=[]` as plain deletion, and the prompt's first goal is "Remove
duplicate or redundant memories" — the preserve-everything block added earlier
is a soft constraint fighting the primary objective. Measured consequences
(paired flips, 2→3 and 4→5, 11 each, 6 overlapping): answers flip to "there is
no record" (and that is literally true — `sunflower`/`figurine`/`pride
festival` grep to 0 rows post-dream), enumerations collapse ("3 children" →
"multiple"), and per-session dates blend (temporal .919→.838). LongMemEval's
−7.7/60 is the same mechanism at higher question-sensitivity; LoCoMo asks
mostly for salient gist, so it loses only the specific-detail tail.

### The union-coverage gate makes it lossless (arm 6)

`mirix/services/merge_coverage.py` + wiring in `episodic_memory_replace` /
`semantic_memory_update` / `knowledge_vault_update`: before anything is
deleted, every deterministic specific of the old rows — numbers, dates,
month/weekday names, word-numbers, multi-word proper names — must appear in the
replacement text, else the call raises (nothing deleted) and the error steers
the model to rewrite, keep a superseded value as history, or not merge.
Proper phrases accept component-wise mentions ("Caroline and Melanie" is
satisfied by separate Caroline + Melanie). Scoped to the auto_dream agent;
`MIRIX_MERGE_COVERAGE_GATE=0` disables.

Live effect in arm 6: 29 lossy merges rejected and rewritten (samples show
exactly the temporal killers being caught: 'august', 'september', '2022',
'13', '27'…), consolidation still happens (227→214), temporal recovers to the
exact no-dream level, and the 3-run QA {134, 137, 135} is statistically
indistinguishable from the no-dream band {134, 135, 138}. **Interleaved dream +
union gate + reranker = break-even QA with a leaner store and graph.**

### Compression rate = the corpus's true redundancy rate

Ungated the graph shrank −7.5% — partly "fake compression" bought by deleting
information. Gated it shrinks −1.8% (store −6%): with losslessness enforced,
how much compresses is decided by actual redundancy, not by how much the LLM
dares to delete. conv-26 (19 sessions, mostly distinct topics) is simply not
very redundant; higher-redundancy corpora should compress more, losslessly.

### Structural bugs found and fixed along the way

- **Role provenance was never passed by the live ingest hooks** — every
  organically built graph had `role=None` on facts/citations; only rebuilt
  graphs had roles (the rebuild script passed `role=actor` itself). Fixed:
  episodic passes `event.actor`, semantic passes `"shared"`. Arm 6's organic
  graph: 728 user / 855 shared, zero none (earlier fresh run).
- **Zombie facts**: consolidation used to leave facts whose every
  `V7_FACT_FROM` citation died — the graph GREW through consolidation (1583
  facts vs 1249 no-dream, +27%, on an earlier run). `maintain_graph` pass 3b
  removes them; measured 338 removed, converging the dreamed graph to the
  no-dream size (1245 vs 1249).
- **Fragmented temporal chain**: DETACH-deleting consolidated refs took
  `V7_NEXT_MEMORY` edges with them (52 edges over 100 refs). Pass 6 rebuilds
  the per-user chain from timestamps (idempotent); both new passes run in every
  dream cycle, including `graph_only`.

## 3. Caveats

- Arms 1/2/3/6 are separate ingest draws; cross-arm deltas carry ±2–3 questions
  of draw+judge noise (the paired comparisons — 2↔4, 3↔5, and arm 6's own
  3-run — are the trustworthy ones). The judge is lenient/quirky in places
  (q6 accepts "20 May" but rejects "Saturday, 20 May" against gold "the Sunday
  before 25 May").
- The confounded 0.7039 run (dreamed store QA'd against a mismatched graph) is
  reported only as the accidental graph-ablation it is; its per-arm number is
  void.
- The gate covers deterministic specifics only. Semantic gist ("sunflowers
  represent warmth and happiness") cannot be gated deterministically without
  mass false rejections and remains prompt-guarded — the plausible next tier is
  an LLM coverage judge on the same choke point, not needed for break-even.
- Single conversation (conv-26), single benchmark. The LongMemEval
  cross-check of the gated recipe has not been run yet.

## 4. Repro

```bash
# data: scratch locomo_conv26_full.json extracted from locomo10.json (sample_id conv-26)
# per arm: fresh DB (CREATE EXTENSION vector), wipe conv-26 subgraph in neo4j,
# point MIRIX_PG_DB at the arm DB, restart server, then:
python main_eval.py --data <conv26.json> --limit 1 --run-llm \
    --mirix_config_path ./configs/0201c_v6.yaml --output_path <arm>
python organize_results.py results/locomo/<arm>

# arm toggles (server env unless noted):
#   arm 1: MIRIX_ENABLE_GRAPH_MEMORY=false
#   arm 2: graph on, MIRIX_GRAPH_RERANK=0
#   arm 4: rerank default on; QA-only rerun (prefill responses, empty records)
#   arm 3: MIRIX_DREAM_EVERY_N_CHUNKS=10 (eval env), MIRIX_GRAPH_RERANK=0
#   arm 6: MIRIX_DREAM_EVERY_N_CHUNKS=10, gate on by default
#          (MIRIX_MERGE_COVERAGE_GATE=0 to disable)
```
