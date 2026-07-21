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
