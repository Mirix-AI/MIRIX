# v7.4–v7.8 + QA failure study — consolidated summary

One clean record of the graph-extractor/resolution line (v7.4–v7.8), the QA-ceiling
diagnosis, and the verified research on how to break the ceiling. All numbers are on
the same 962-memory LongMemEval-S store (`mirix_lm114_pm`, 632 episodic + 330 semantic)
unless noted, so cross-version QA is a clean same-memory comparison.

## 1. The version line at a glance

| ver | change | extract /mem | build 962 mem | anchors | singleton | QA (/60) | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| v7 | LightRAG extractor + `anchor_canonical_key` (surface dedup) — **baseline** | 15.25s | ~24min (×10, extrap.) | 5165 | 66% | 30 | baseline (post-merge full) |
| **v7.4** | **GLiNER** encoder extractor (local, deterministic) | **0.09s (~170×)** | **1m38s** (×10) | 2473 | 66% | 32 | **superseded** — fast + kills User/Assistant hubs, but a span tagger **can't abstract** → drops LightRAG's concept anchors (networking/team-collab = 0); concept-heavy retrieval degrades (masked by net QA) |
| **v7.6** | **LLM triple extraction (direction D)** — abstracts + keeps relations as `V7_RELATION` edges | 5.9s (2.6×) | **9m41s** (×10) | 5777 | 81% | 31 | **foundation** — concept anchors back (41% concept), **3667 relation edges** (v7 had none); real value = relations, not speed |
| **v7.7** | **PPR retrieval** over v7.6's relation graph (networkx, no GDS) | — | — (retrieval 1.6s/query) | — | — | 30 | retrieval **much** better (colleague query: fitness→social; context 500→37k chars) but **QA flat** → **pivotal finding** |
| **v7.8** | **registry-guided entity resolution** — embedding candidates + **LLM verify-merge**, rename to canonical at extract | ~9s (5.9 + resolve) | **36m08s** (×4; ≈14min if ×10) | 4744 (−18%) | 66% | **34 (best)** | cleaned the redundancy v7.6 introduced; **first change to move QA** (+3 vs v7.6; re-run to confirm vs answerer-LLM noise) |

*Build times measured on the 962-memory rebuild; "×N" = the `rebuild_graph.py` concurrency
used (v7.8 was run at 4 for registry consistency, so its 36min ≠ apples-to-apples with the
×10 runs — per-memory it's ~9s, i.e. ~14min at ×10). v7 LightRAG's 962-build is extrapolated
from its measured 15.25s/mem (no clean full rebuild was run at that cost).*

(Earlier: v7.1 rerank, v7.2 coverage — retrieval tweaks; v7.3 proposition ingest — rejected, collapsed edge structure. v7.5 role/domain scoping — planned, not built.)

QA line: **v7 30 · v7.4 32 · v7.6 31 · v7.7 30 · v7.8 34.**

## 2. Key findings per version

- **v7.4 (GLiNER) — the "faster but can't abstract" lesson.** ~170× faster and removes the
  User/Assistant noise mega-hubs, but a span tagger only tags literal spans. v7's concept
  anchors ("Networking", "Team Collaboration") were LightRAG **abstractions** — the words
  aren't in the text — so GLiNER cannot reproduce them (verified: aggressive tuning can't
  recover them; hard limit). Kept only as a cheap named-entity supplement.
- **v7.6 (D) — LLM triples, the SOTA-aligned extractor.** Every recent top-venue KG method
  (HippoRAG 2, KGGen, AutoSchemaKG, EDC) uses LLM triple extraction; GLiNER is the outlier.
  D restores abstraction **and** keeps the relations v7 discarded (→ `V7_RELATION` edges,
  the substrate for multi-hop). Corrected speed claim: only **2.6×** faster than LightRAG
  (I first said 34× by comparing concurrent-D against sequential-LightRAG — an error).
- **v7.7 (PPR) — the pivotal finding.** PPR made retrieval clearly better, yet QA stayed
  flat across all four graph variants (30–32). Root cause, verified from transcripts:
  **the answerer calls its own `search_memory` (flat pgvector) on 60/60 questions**, so the
  graph context we inject is only a *secondary* source — the parallel flat search dominates.
  Retrieval/graph-context quality is **not** the QA bottleneck.
- **v7.8 (entity resolution) — the redundancy fix, and it moved QA.** v7.6's LLM naming was
  inconsistent → 81% singleton, ~30% near-duplicate anchors ("Outward Hound Brick Puzzle"
  vs "…'s Brick Puzzle", cos 1.00). Surface `anchor_canonical_key` can't fix this and
  embedding-cosine merge can't decide identity (Monday≈Tuesday, 42≈46 campsites are close
  but distinct). v7.8 = **embedding finds candidates, an LLM decides same-or-new** (EDC/KARMA
  style). Result: anchors −18%, singleton 81%→66%, **no over-merge** (numbers/days kept
  apart), and QA 34 (best). This is the paper's canonicalization layer — and the only
  mechanism that can also *split* same-string entities (G1), because the LLM has the names.

## 3. Why QA is stuck ~30/60 — the ceiling diagnosis

23 questions fail across **all** retrieval variants (systematic ceiling; the other ~13 that
differ between versions are answerer-LLM noise, no consistent graph-strength pattern). Each
of the 23 was judged against the retrieved evidence **and** the raw memory store:

| failure category | # | who can fix it |
|---|---:|---|
| **reasoning error** — correct evidence retrieved, wrong answer (evidence "American Airlines" → answered "JetBlue"; memory "50 hours" → "45"; over-committed on an abstention question) | 5 | **answerer only** |
| **counting / aggregation** — instances exist & are retrieved but the LLM under-counts (3 of 4 art events; 1 of 3 jewelry; $2000 of $2500) | 5 | answerer (enumerate-then-count); graph enumeration as a completeness aid |
| **preference synthesis** — "suggest X for me" needs the user's history synthesized; gets generic advice | 4 | answerer + a persona store |
| **extraction miss** — the fact was never written to memory ("500 Mbps" = 0 memories; "7 shirts" = 0) | 4 | **write-time extraction only** — unrecoverable by any retriever/graph |
| **temporal** — date arithmetic / event ordering | 3 | temporal index + code date-math |
| **retrieval miss** — fact in memory but not surfaced | ~2 | (nearly none) |

**Bottom line: ~61% answerer-side, ~17% extraction, ~0% pure retrieval.** No graph/retrieval
change can move the answerer/extraction failures. (Illustrative: the "5 fitness classes"
miss — the 5th is yoga, which is *all over* memory; the answerer just didn't count Sunday/
app yoga as a "class." Aggregation reasoning, not fragmentation or retrieval.)

## 4. QA research — how to break the ceiling (verified, 2024–2026)

The strongest external anchor: **LongMemEval (ICLR 2025) itself shows the bottleneck is
reading/reasoning, not retrieval**, and its own fix (Chain-of-Note reading) gives up to
**+10pp**. And **Emergence AI's independent LongMemEval harness: SOTA 86% = session-granular
retrieval + rerank + a reasoning read step — NOT a fancier graph** (graph systems like Zep
sit at 71%). Preference is the hardest category even for SOTA (60%).

| our failure mode | fix (method @ verified venue) | how it plugs into "answerer + search_memory" |
|---|---|---|
| reasoning error (5) | **Chain-of-Note** (EMNLP'24; LongMemEval-native, +10pp) → cite-memory-id + **Self-RAG ISSUP** self-check → **CoVe** (Findings ACL'24) for the hardest | per-memory "reading notes" before answering + a grounding self-check pass |
| counting (5) | **enumerate-then-count**: force a JSON list of every supporting memory, then **PAL/PoT** (ICML'23) counts with code | new count/aggregate tool + prompt; graph `enumerate_entity` as completeness backend |
| abstention (subset of reasoning) | **Sufficient-Context gate** (ICLR'25): "is this enough to answer? if not → not-enough-info". ⚠️ **AbstentionBench (NeurIPS'25): CoT *hurts* abstention** → need an explicit gate, not more reasoning | post-retrieval LLM-judge gate that can emit abstain |
| preference (4) | **persona/preference store**: LaMP-PAG (ACL'24, minimal) / PersonaAgent (NeurIPS'25) | ingestion-time preference extraction into a persona sub-store; **inject** the profile for advice-shaped questions (don't rely on search) |
| extraction miss (4) | **Dense-X propositions** (EMNLP'24) + **SeCom** segment fallback (ICLR'25) + LongMemEval fact-key expansion | write-time: atomic-proposition extraction so cold numerics each get a row (reconnects the v7.3 line) |
| temporal (3) | **TReMu** (Findings ACL'25): resolve relative→absolute dates at ingest + **code** date math; **bi-temporal Neo4j edges** (Zep) | ingest stores absolute dates; answerer gets a date-math tool; graph edges carry valid-time |

**Cross-cutting:** one "read → structure → verify" answerer stage (Chain-of-Note JSON notes)
feeds counting (enumerate-then-count), grounding (cite + ISSUP), and abstention (sufficiency
gate) at once — hits ~10 of the 23 (reasoning 5 + counting 5), cheapest (prompt/tool, no
re-ingest), and CoN is the only method with a published LongMemEval result. **Start there.**

## 5. Where we are / next

- **Graph line (v7.4–v7.8): done and committed.** v7.6 (relations) is the foundation; v7.8
  (entity resolution) is the clean version and the paper's canonicalization contribution, and
  it moved QA (34, best — confirm with a re-run to rule out answerer-LLM noise).
- **The QA lever is the answerer, not the graph.** To go past ~34: (1) Chain-of-Note +
  enumerate-then-count + sufficiency gate (biggest, cheapest — ~10 questions); (2) proposition
  extraction for the 4 unrecoverable extraction misses; (3) persona store for preference;
  (4) temporal absolute-dates + code math.
- **Graph's genuine role going forward:** structured **tools** for the answerer (an
  `enumerate_entity` completeness backend for counting; bi-temporal edges for ordering; a
  persona subgraph) — expose the graph as tools the answerer calls, not context it ignores.

### Commits
v7.4, v7.6, v7.7 and v7.8 all land in `pre-squash tag` (the extractor-line commit) — the
branch history was consolidated into six thematic commits, so a version no longer
owns a commit one-to-one. The original per-version commits (`73c5207` v7.4,
`7fb68fc` v7.6, `dcefe3e` v7.7, `a04de8f` v7.8) are preserved at the tag
`graph_revision_pre_squash`; check that out to get a single version's exact code
state. Branch `graph_revision`, ahead of `origin/main` (unpushed).
