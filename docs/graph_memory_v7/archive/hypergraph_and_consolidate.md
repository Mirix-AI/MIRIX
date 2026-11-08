# Hypergraph (V7Fact) + `consolidate` answerer tool — build log & negative result

Two experiments on the same 962-memory LongMemEval-S store (`mirix_lm114_pm`,
632 episodic + 330 semantic). Both are **kept in the tree but rejected as
defaults**; the reasons are the scientifically useful part.

> Numbering note: these were built as "v7.10 (hypergraph)" and "v7.11
> (consolidate)" during the session. That collides with `version_ledger.md`'s
> *roadmap* entries (v7.10 = LLM verify-merge gate, v7.11 = schema induction).
> The roadmap is unchanged; this doc records what was actually built/measured.

## 1. Hypergraph — reify each triple as a `V7Fact` hyperedge

`graph_memory_manager_v7._upsert_facts` (fires when `graph_version == "v7.10"`):
each extracted triple becomes a `V7Fact` node carrying `predicate`, `role`
(user/assistant/shared), `timestamp`, linked by `V7_FACT_SUBJECT` /
`V7_FACT_OBJECT` / `V7_FACT_FROM` to the subject anchor, object anchor, and
source memory. Built store: **4785 anchors + 4633 V7Fact** over 962 memories.

- The structure is sound and is the paper's "facts as first-class, role- and
  time-stamped hyperedges" contribution.

### Refinement pass (`evals/refine_hypergraph.py`) — redundancy removed, free

The as-built hypergraph carried real redundancy. Four passes, run to convergence
(a second pass finds nothing), **at zero QA cost**:

| | before | after | |
|---|---:|---:|---|
| V7Fact | 4,633 | **4,306** | −7.1% |
| V7Anchor | 4,785 | **3,665** | −23.4% |
| distinct predicates | 1,320 | **1,221** | −7.5% |
| edges | 26,639 | **24,713** | −7.2% |

1. **Tautologies (112)** — extraction artifacts where subject == object
   (`french -include-> french`). Deleted.
2. **Duplicate triples (215)** — identical (subject, predicate, object) collapsed
   to ONE fact node, with each original source re-attached as an extra
   `V7_FACT_FROM` edge. **Provenance is preserved, not lost**: 131 facts now cite
   >1 memory (max 9), e.g. *"Seven Husbands of Evelyn Hugo -written by-> Taylor
   Jenkins Reid"* cited by 4 memories. This is the correct hypergraph semantics —
   a fact is one fact, cited N times, not N copies.
3. **Predicate canonicalization (99 variants)** — `includes`(421) folded into
   `include`(→823); `is/are located in` → `located in`; `offers` → `offer`.
4. **Dead-weight anchors (1,120)** — no fact touches them *and* they link ≤1
   memory, so they can neither answer nor bridge. Pruned. (Consistent with the
   earlier v8 result: −66% anchors at zero accuracy cost.)

Post-refinement invariants verified: 0 tautologies, 0 duplicate triples, 0
malformed facts (every fact keeps subject + object + ≥1 source), 0 orphan
anchors. **QA re-checked on the refined graph: 36/60**, inside the cold-fact band
(37/36/37), with all five recovered-fact wins (Q46/Q56/Q57/Q1/Q15) still passing.

### Prevented at ingest (3 of the 4 no longer need cleanup)

`_upsert_facts` was *generating* the redundancy: it minted a random `gen_id()` per
extraction, so `MERGE (f:V7Fact {id: ...})` always CREATED — the same triple from
N memories became N nodes. It also stored the raw predicate surface form and had
no tautology guard (while `_link_relation_edges` right below it did). Fixed at
write time:

- **Deterministic identity** — `fact_identity(user, subj_key, predicate, obj_key)`
  hashes the canonical triple, so the same assertion MERGEs to ONE fact and each
  new memory only adds a `V7_FACT_FROM` citation edge.
- **Canonical predicates** — `canon_predicate()` folds copulas and singularizes
  the verb (`includes`→`include`, `is located in`→`located in`) before storage.
- **Tautology guard** — `sk == ok` triples are dropped, matching the existing
  relation-edge guard.
- **Role/time moved to the citation** — they are properties of *this memory
  asserting the fact*, not of the fact, now that one fact spans memories. The
  node keeps first-seen values via `ON CREATE SET` (nothing else read them).

Verified end-to-end: the same triple submitted from two memories with different
surface forms plus a tautology yields **1 fact node**, predicate `include`, cited
by both memories with per-citation roles (`user`, `assistant`).

**The 4th (dead-weight anchors) is not preventable at write time** — whether an
anchor ever gets a fact or a second memory is a corpus-global property unknown
when it is created. That one stays a periodic maintenance pass
(`refine_hypergraph.py`, pass 4, and `maintain_graph` in the auto_dream cycle).

### Verified by a full rebuild — clean from birth

Rebuilding all 962 memories with the new write path, then checking **without
running any cleanup**:

| check | result |
|---|---|
| tautologies | **0** |
| duplicate triples | **0** |
| `includes`-style predicate variants | **0** |
| citation edges carrying their own role | **4,471 / 4,471 (100%)** |

Final shape: 4,305 facts · 4,862 anchors → **3,689 after** the maintenance pass,
which reported `{tautologies: 0, duplicates: 0, dead_anchors_pruned: 1,173}` —
exactly the intended division of labour: ingest prevents three kinds, the periodic
pass collects the one it cannot. 137 facts are cited by >1 memory (max 14), each
citation keeping its own role (shared 2,619 / assistant 1,048 / user 804).

**A real bug the dedup change introduced.** Deterministic ids mean concurrent
ingests MERGE onto the *same* node, so they contend for its lock — a rebuild at
concurrency 10 started failing with `Neo.TransientError.Transaction.
DeadlockDetected`. The old random-id scheme never collided, so this only appears
once dedup actually works, and only under concurrency: a serial test cannot catch
it. Fixed by sorting rows by fact id (uniform lock-acquisition order) plus
exponential-backoff retry on transient errors. Re-verified: 962 memories at
concurrency 10, **0 failures**.

### How this interacts with auto_dream

`auto_dream` is the PG-side analogue of this work: an LLM agent that reviews
memories for duplicates/overlaps/conflicts and merges them via
`episodic_memory_replace` / `semantic_memory_update` (prefer merging over
deletion; keep the uncertainty when a conflict is unresolvable).

Tracing that path matters for graph coherence:
- `episodic_memory_replace` **hard-deletes** the old rows and touches nothing in
  the graph → dangling refs. This is what `maintain_graph`'s orphan sweep exists
  to clean.
- It then **re-inserts** the merged item through `insert_event`, which *does* call
  `process_memory` → the merged memory gets fresh graph refs.

So the graph stays coherent across a dream cycle, provided the sweep runs. Note
the standing assumption: `process_memory` is wired only to the `insert_*` paths,
so any future in-place memory update would silently leave the graph stale.

(auto_dream has never run on the eval store — 0 checkpoints — so none of the
measurements in these docs are affected by it.)
- The **`query_facts` answerer tool** over it was **rejected**: QA 36 → 32. It
  returned descriptive/off-topic facts that misled the answerer. Superseded by
  `consolidate` (below), which is a better mechanism but also nets out negative.

## 2. `consolidate` — exhaustive distinct-instance enumerator (answerer tool)

Root-cause hypothesis it was built to fix: countable *personal* instances (the 4
bikes, 5 fitness classes) are **scattered across memories and never aggregated**,
so top-k flat search under-counts.

`TaskAgent._consolidate(topic)`: embed topic → (a) graph anchor→memory recall +
(b) pgvector recall over memory `summary_embedding`, union → fetch full PG text →
LLM de-dup into `{items, count}`. Three real bugs were found and fixed along the
way — each is a reusable lesson:

1. **Embedding-model mismatch.** MIRIX's `embeddings.embedding_model()` never
   passes `config.embedding_model` to llama-index's `OpenAIEmbedding`, so it
   silently defaults to **`text-embedding-ada-002`**, not the configured
   `text-embedding-3-small`. Both are 1536-dim, which hid it. The **entire graph
   + PG store is ada-002**; a raw 3-small query lands in a different space
   (anchor match cos 0.52 vs 0.94). Any hand-written retrieval MUST use ada-002.
2. **Detail truncation.** Enumerations live deep in `details`
   ("...four bikes: a road bike, mountain bike, commuter bike, and a new hybrid
   bike") — truncating details to 300 chars dropped the count.
3. **Graph-only recall is incomplete.** The "four bikes" memory was anchored
   under trip *locations* (San Francisco, Jackson Hole), never linked to a *bike*
   anchor — an extraction-linking gap. Adding a pgvector recall channel
   (zero-padded to the store's 4096-dim column) recovered it.

**Mechanism validated** on hand-picked topics: bikes 4/4, art 4/4, jewelry 3/3,
fitness 5/5.

## 3. End-to-end QA — rejected

| answerer config | QA /60 |
|---|---:|
| enumerate-then-count (baseline) | **36** |
| + consolidate, trust its count | 30 |
| + consolidate, scoped to distinct-instance + advisory | 31 |

Both variants regress. Gated behind `MIRIX_ENABLE_CONSOLIDATE` (default off =
the 36 answerer).

## 4. Why it can't help — 3-run consensus decomposition

Scoring three full runs (36 / 30 / 31) per-question and partitioning:

| partition | # of 60 | meaning |
|---|---:|---|
| stably correct (1/1/1) | 23 | always answered right |
| **stably wrong (0/0/0)** | **17** | **the true ceiling** |
| **flips across runs** | **20** | **answerer-LLM noise** |

**Layer A — the metric is noise-dominated (biggest cause).** Score = 23 +
(hits on the 20 unstable). Floor 23, ceiling 43; 36/30/31 = 23 + {13, 7, 8}.
**"36" was a lucky draw on the unstable set, not a real win.** Any answerer
change smaller than ≈±6 is undetectable in a single run. This retroactively
explains the whole answerer line (v7.9 34→30, v7.10 `query_facts` 36→32,
v7.11 36→30/31): **all within one noise band — we were chasing noise.**

**Layer B — consolidate's addressable set is tiny and qualifier-gated.** The
distinct-instance counting questions it targets (Q17 art, Q26 fitness, Q59
jewelry, Q54 devices) all carry temporal/scope qualifiers — "in the past month",
"in a typical week", "in the last two months", "in a day" — that a flat topic
enumerator ignores. The hand-tests looked perfect only because they *dropped the
qualifier*. And those questions sit in the noise band or the ceiling anyway.

**Layer C — the true ceiling isn't an enumeration problem.** The 17 stably-wrong:
cross-session aggregation (7), preference synthesis (4), quantity-sums (fish=17,
luxury total=$2500, discount %), stored-value/extraction-miss (50 hours, 500
Mbps, 7 shirts). **None** is "scattered distinct instances need enumerating" —
consolidate's target mode is essentially absent from the ceiling.

## 5. Implications (what to actually do next)

1. **Validate answerer changes with ≥3 runs / averaging.** Single-run deltas of
   ±6 are noise. Every prior "version" QA number is a noisy point estimate.
2. **Attack the stable ceiling, not counting.** Cross-session aggregation (7) +
   preference synthesis (4) = 11 questions, 65% of the ceiling. That is the real
   target for the paper, not distinct-instance enumeration.
3. Keep the hypergraph as structure and `consolidate` as an A/B-able tool; give
   consolidate temporal/scope filtering before re-testing it.
