# v7 Graph Memory — Version Ledger

Each **successful** change to the graph pipeline mints one version (v7.4, v7.5, …).
A change is "successful" only if it beats the baseline on its *target* metric
**without regressing QA beyond the noise floor**. Failed experiments are recorded
here with a `rejected` verdict but do **not** consume a version number (precedent:
the v8.1 User/Assistant role-strip was net-negative and never became a code
version).

## How a version is minted (checklist)

1. Implement behind the existing switch — add the version string to the
   `graph_version` guard tuple in `graph_memory_manager_v7.py::process_memory`
   and `graph_retriever_v7.py` (currently `("v7","v7.1","v7.2","v7.3","v8")`).
2. Ingest the **same first 10 LongMemEval-S chunks** into an isolated DB
   (`mirix_lm10_<ver>`), graph wiped first.
3. **Tier-1 (cheap, on 10-chunk)** — record vs baseline: extraction time / LLM
   calls, anchor count, singleton %, hub composition, new-edge counts. Plus the
   **G1/G2 probe** score where relevant.
4. **Tier-1 gate**: effect must exceed the **noise floor** (see below) and show
   no structural regression.
5. **Tier-2 (expensive, only if gated)** — full 114-chunk LongMem-S (60 QA) +
   MH-Doc + SH-Doc as applicable. QA judged by the MAB judge.
6. If it wins: `git commit` (`feat: v7.x <change>`), save a DB+graph snapshot
   under `~/MIRIX_eval/saved/lm10_<ver>/`, add a ledger row.
   If it loses: add a ledger row with `rejected` + the reason; revert the guard.

## Noise floor

Extraction is LLM-based and non-deterministic (the meta-agent's per-chunk routing
and each sub-agent's yield both vary). Measured over **5 fresh post-merge 10-chunk
builds** (2026-07-18, code 7524139): a delta smaller than this spread is **not** a
real effect.

| run (10-chunk, post-merge v7) | ep+sem memories |
|---|---:|
| nf1 / nf2 / nf3 / nf4 / nf5 | 93 / 79 / 118 / 106 / 78 |
| **mean ± sd** | **95 ± 15.5** |
| **spread (min–max)** | **78–118 = 42% of mean** |

Reference: baseline #1 (pre-merge) = 116; baseline #2 (post-merge) = 58 (a low
outlier). The 5 post-merge runs (78–118) bracket the pre-merge 116 → the merge had
**zero** effect on extraction; the 116-vs-58 gap was pure noise.

**Consequence for the eval plan**: run-to-run memory volume swings ±42%, so a
single 10-chunk build **cannot** detect a change smaller than that. Anchor count
scales ~linearly with memories, so anchor-count deltas inherit the same ~42% floor.
→ 10-chunk intrinsic counts are only usable for **large** effects (v7.4's User/
Assistant hub removal, extraction-time win); subtle changes (disambiguation) must
be measured on the deterministic **G1/G2 probe**, not 10-chunk anchor counts.
(Anchor/singleton for the 5 runs was lost to a `measure` bug — missing `AS` in the
Cypher — so only PG-side ep+sem survived; fixed in `overnight_c.sh`.)

### Full 114-chunk post-merge v7 baseline (2026-07-18, `mirix_lm114_pm`)

Real QA reference for v7.4+. 962 memories (632 ep + 330 sem), 5165 anchors, 66%
singleton. **QA = 30/60 (50%)**, judged by MAB judge.

| category | acc | | category | acc |
|---|---:|---|---|---:|
| knowledge-update | 8/9 (89%) | | temporal-reasoning | 7/15 (47%) |
| single-session-assistant | 5/6 (83%) | | **multi-session** | **4/15 (27%)** |
| single-session-user | 5/9 (56%) | | single-session-preference | 1/6 (17%) |

**multi-session (multi-hop) = 27% is the weakest** — exactly the target of v7.5 (PPR)
and v7.7 (disambiguation). (Pre-merge full run was 35/60; this 30/60 is a low
extraction draw within the ±42% noise, not a regression.)

## Version map

| ver | change | targets gap | status | anchors | singleton% | QA (full / MH / SH) | commit |
|---|---|---|---|---:|---:|---|---|
| v7 | LightRAG extractor + `anchor_canonical_key` (surface dedup) | baseline | **current** | 5165 (full) | 66% | **30/60 (post-merge full)** | 7524139 |
| v7.1 | rerank candidates by query text-cosine | retrieval | shipped | — | — | SH-Doc +4 | 5484a8d |
| v7.2 | per-anchor coverage round-robin | retrieval | shipped (neutral) | — | — | +0 | 5484a8d |
| v7.3 | proposition ingest, no LightRAG | G6 / extraction | **rejected** — collapsed edge structure (only DESCRIBED_BY survived; no episodic → lost APPEARS_IN / SUPPORTED_BY / NEXT_MEMORY); singleton ↑77% | 439 | 77% | not run (structure not comparable) | 0a7e7f1 |
| **v7.4** | **GLiNER extractor** (local encoder) — dialogue roles (User/Assistant) filtered out | **G6** cost + **G4** noise hubs | **shipped-but-superseded** — ~200× faster, hubs gone, deterministic; **BUT GLiNER can't abstract → drops LightRAG's concept anchors (networking/team-collaboration=0), degrading concept-heavy retrieval (masked by net-neutral QA). See `extractor_direction_D.md`.** Kept as a cheap named-entity supplement. | 2473 (−52%) | 66% | **32/60 (vs v7 30/60; concept-loss caveat)** | 73c5207 |
| **v7.6** | **LLM triple extraction (direction D)** — abstracts concepts (GLiNER can't) + keeps relations as `V7_RELATION` anchor→anchor edges; User/Assistant filtered | G6 cost + concept coverage + **relations (for PPR)** | **foundation shipped** — concept anchors back (2374, 41% vs v7.4's 294/12%), **3667 relation edges** (v7 had none), hubs gone, **34× faster than LightRAG** (9.7min vs ~5.5h for 962 mem). QA **31/60** = flat (v7 30 / v7.4 32, all within ±42% noise) **BY DESIGN: retrieval doesn't traverse V7_RELATION yet — the payoff needs PPR (next).** | 5777 | 81% | 31/60 (foundation; PPR unlocks relations) | pending |
| **v7.7** | **PPR retrieval** over the v7.6 relation graph (query→anchor seed + Personalized PageRank, networkx — no GDS) | multi-session (27%) | **implemented; retrieval much better, QA flat (30/60)** — the colleague query reversed from off-topic fitness to on-topic social; context went from ~500-char titles to 37k-char rich. But QA = v7 30 / v7.4 32 / v7.6 31 / v7.7 30, **all flat.** | — | — | 30/60 | pending |

### ⚠️ Pivotal finding: graph-context quality is NOT the QA bottleneck

Four retrieval variants (v7 anchor-search, v7.4 GLiNER, v7.6 D, v7.7 PPR) span a
huge range of graph-context quality yet **all land 30–32/60**. Root cause, verified
from the QA transcripts: **the answerer calls its own `search_memory` (flat pgvector)
tool on 60/60 questions.** The graph context we inject via `wrap_user_prompt` is only
*one of two* retrieval sources — the answerer's parallel flat search does the heavy
lifting, so improving the graph context barely moves QA.

Implication — the whole v7.4→v7.7 line optimized a **secondary** path. To make the
graph move QA it must either (a) provide what flat search cannot (true multi-hop
bridges) AND be surfaced so the answerer uses it, or (b) *replace* the answerer's flat
search (the earlier "graph-routed search" did this and was −18pp — flat search is a
strong baseline). Next investigation should target the answerer's retrieval path, not
graph-context quality. The v7.6 relation graph + v7.7 PPR remain a sound foundation
*if* retrieval is rerouted; as a supplement to flat search they are ~neutral.
| **v7.5** | **role/domain scoping** — User/Assistant become a `role ∈ {user,assistant,shared}` scope attribute on memory refs (from provenance, NOT an entity anchor); retrieval can filter by role. Built on v7.4. | single-session-user / assistant / **preference** (17%) | planned | — | — | — | — |
| **v7.6** | keep relations + PPR retrieval (query→triple + passage-seeded) | multi-hop retrieval | planned (enables 7.7/7.8) | — | — | — | — |
| **v7.7** | synonym edges (`name_embedding` cos ≥ τ, edge not merge) | **G2** alias, **G4** | planned | — | — | — | — |
| **v7.8** | accumulated registry + embedding candidates + **LLM verify-merge gate** | **G1 + G2 + G3** | planned — **paper core** (only mechanism that can *split*, i.e. touch G1) | — | — | — | — |
| v7.9 | schema induction (AutoSchemaKG-style conceptualization) | G5 | deferred | — | — | — | — |

Ordering notes:
- **v7.5 role/domain** reframes the User/Assistant problem: v7 makes them mega-hub
  anchors (deg 452/233, a category error — a dialogue role is not an entity); v8.1
  *deleted* them and was net-negative (lost aggregation value). v7.5 keeps the
  signal as a scope attribute instead. Targets the benchmark's user/assistant/
  preference categories directly.
- **v7.6 (retrieval/PPR) ships before v7.7/v7.8** because synonym/resolution edges
  are inert until retrieval traverses the graph (PPR). Until then, evaluate
  v7.7/v7.8 on the G1/G2 probe (intrinsic), not on QA.

## G1/G2 probe set

A small labelled set that measures disambiguation *independently of retrieval*
(end-to-end QA cannot isolate it). Seeded from the SH-Doc same-name breaks.

- **G1 (same surface name → different real entity)**: `(name, context, gold entity)`;
  metric = were they wrongly merged into one anchor? e.g. `Margaret` (Thatcher vs
  Margaret of Scotland), `naval base` (Kings Bay vs Dyrrachium).
- **G2 (one entity → different names)**: `(name_a, name_b, same?)`; metric = were
  they linked? e.g. `Thatcher` ↔ `Margaret Thatcher`.

Stored at `~/MIRIX_eval/probes/g1g2_probe.jsonl` (see `build_probe.py`).
