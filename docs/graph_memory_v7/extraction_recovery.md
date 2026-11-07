# Cold-fact recovery — the first real QA gain (+4.4)

The answerer-side result line for LongMemEval-S (60 QA, `mirix_lm114_pm`, 962
memories). After establishing that the QA metric is noise-bound and that the
failure floor is **write-time fact-loss**, a selective cold-fact recovery pass
gives the project's first reproducible QA improvement.

## 1. The measurement is noise-bound — validate with multi-run

Running the **same** answerer config (enumerate-then-count) four times: **36, 30,
33, 30** — mean **32.2**, sd **2.5**, 33% of questions (20/60) flip run-to-run.
Any single-run delta < ~±6 is noise. Every prior "version" QA number (v7 30, v7.4
32, v7.6 31, v7.7 30, v7.8 34, enumerate 36) is a draw from this band — they are
statistically indistinguishable. **All answerer/graph comparisons must be
multi-run.**

## 2. The failure floor is write-time fact-loss, not reasoning

3-/4-run consensus splits the 60 into ~22 stably-correct, ~18 stably-wrong (the
true ceiling), ~20 noise. The stably-wrong floor is dominated by facts the
**summarizing ingest dropped**. Verified against the raw HF source — the exact
value is present verbatim but absent from the stored memory:

| Q | gold | in raw source |
|---|---|---|
| Q46 internet | 500 Mbps | "upgraded to **500 Mbps** about three weeks ago" |
| Q56 shirts | 7 | "brought **7 shirts** and 5 pairs of shorts" |
| Q1 work hours | 50 | "working up to **50 hours per week**" |
| Q15 cameras | 3 months | "collecting vintage cameras for **three months**" |
| Q57 farmers | $420 | "earned **$420** at the Downtown Farmers Market" |

MIRIX's multi-agent ingest writes *summaries* ("User discussed packing for Costa
Rica") and drops the peripheral specifics. This is the Dense-X / LongMemEval
"summaries lose cold facts" problem, confirmed on our own store.

## 3. What did NOT work (all multi-run, all gated-off)

Three answerer-side interventions, each targeting a slice of the floor — **all
land at or below 32.2**, because the floor is retrieval/extraction, not answerer
reasoning, and adding context perturbs the 20 noisy questions:

- **v7.11 `consolidate`** (distinct-instance counting tool): 30/31. Mechanism
  works on hand-picked topics but the benchmark's counting Qs are mostly
  quantity-sums it can't help; wins already captured by enumerate-then-count.
- **temporal prompt** (use given Current Date, exact-date filter, explicit
  subtraction): 31. The "temporal" failures are actually **retrieval/grounding**
  (Q48 retrieved the wrong workshop date; Q29 confused considered-vs-flew) — a
  date-math prompt can't fix a fact it can't find.
- **persona store** (LaMP-PAG-style profile injected for advice Qs): 30. Barely
  moved its targets (the profile lacked the exact details — "lemon poppyseed",
  "30-day challenge" — because *those were summarized away too*) and broke
  already-passing advice Qs.

## 4. What worked — selective cold-fact recovery + search-merge (+4.4)

Two ingredients:

1. **Selective cold-fact extraction** (`evals/build_coldfacts.py`): a second pass
   over raw USER turns, keeping only turns bearing a literal (digit / price / % /
   **word-number** like "three months") — most turns emit nothing, so no Dense-X
   row explosion. An LLM turns each into a self-contained fact preserving the
   verbatim value ("The user upgraded to 500 Mbps", "…collecting cameras for
   three months"). 129 facts, embedded with **ada-002** (the model MIRIX's ingest
   actually uses — `embedding_model()` silently defaults llama-index to ada-002,
   never passing `config.embedding_model`; the whole store is ada-002).
2. **Search-merge, not injection** (`TaskAgent`, gated `MIRIX_COLDFACT`): cold
   facts matching the answerer's *search query* are appended to `search_memory`
   results **as regular retrieved evidence** — not force-injected as "verified
   ground truth". This is the load-bearing design choice.

**Injection vs merge — the decisive comparison (both recover the same facts):**

| | mean | note |
|---|---|---|
| baseline | 32.2 | — |
| cold-fact **injection** (system prompt, "treat as ground truth") | ~31.7 | flat — flips the 4 targets but "ground truth" framing *overrides correct answers* (Q36/Q40: 4/4→0/3) and conflicting facts mislead (Q16) |
| cold-fact **search-merge** (regular evidence) | **36.7** | **+4.4** — same target wins, but retrieval-competition framing avoids the over-trust collateral |

Force-injection and merge recover the *identical* facts; the only difference is
whether the answerer treats them as commands or as evidence. Evidence wins.

**Result (search-merge + refined extractor), 3-run:** **37 / 36 / 37 — mean 36.7,
sd 0.5.** vs baseline 32.2 / sd 2.5. The distributions barely overlap (cold-fact
min 36 = baseline max 36), and variance collapses because the wins are
**deterministic** (the recovered fact flips the same question every run).

Per-question: **+7 won / −3 lost = +4 net deterministic.**
- Won: Q46/Q56/Q57/Q1/Q15 (0→3/3, previously impossible — the fact wasn't in
  memory), plus Q48/Q4.
- Lost: Q16 ($800 vs $1,200 conflicting handbag prices), Q28 (both "three bikes"
  and "four bikes" facts collide), Q33 (ordering). Residual collateral =
  conflicting/ambiguous cold facts; future work = dedup-by-recency + skip
  ordering/abstention Qs.

## 4b. Rebuilding the graph adds another +4.0 (best: 40.7)

Rebuilding the hypergraph with the clean write path (deterministic fact ids,
canonical predicates, tautology guard, per-citation roles) and re-running the same
cold-fact answerer:

| config | runs | mean | sd |
|---|---|---:|---:|
| baseline (no cold-fact) | 36 / 30 / 33 / 30 | 32.2 | 2.5 |
| cold-fact + old graph | 37 / 36 / 37 | 36.7 | 0.5 |
| **cold-fact + rebuilt graph** | **39 / 42 / 41** | **40.7** | 1.2 |

**+8.5 over baseline**, and the three distributions do not overlap (rebuilt min 39
> old-graph max 37), so this is not noise. All five recovered-fact targets still
pass.

The gains (+7 / −2 vs the old graph) cluster tellingly in **temporal-ordering**
(Q25 Spanish-classes 1/3→3/3, Q33 Page-Turners 0/3→3/3, Q39 camping-days 0/3→2/3)
and **multi-session aggregation** (Q17 art-events 0/3→3/3, Q59 jewelry 0/3→2/3),
plus two recall questions (Q52, Q42). That is precisely where graph structure —
the temporal chain and cross-memory links — should help, and it is the **first
time in this project that graph work moved QA at all**, contradicting the earlier
"retrieval/graph is not the bottleneck" finding.

⚠️ **Attribution is not isolated.** The rebuild changed two things at once:
(a) structure (dedup, canonical predicates, per-citation roles) and (b) content
(a completely fresh, non-deterministic LLM extraction). This experiment cannot
separate them — the +4.0 could be largely a luckier extraction draw. The
consistency of the temporal flips (0/3 → 3/3, not a scattered 1-or-2) *suggests* a
structural cause, but suggestion is not proof. The controlled test is to rebuild
once more with the **old** write path and re-run 3× QA; until then this number
should be reported as "rebuilt graph", not "clean graph caused it".

## 5. Takeaways

- **The lever is write-time, not the answerer.** Recovering summarized-away facts
  is the only intervention that beat the noise band — and rebuilding the graph
  (also write-time) added the next +4.0. Every gain in this project came from what
  gets *written*, never from how the answerer was told to *read*.
- **Provide recovered facts as evidence, not ground truth.** Same facts, +5
  swing between the two framings.
- **Deterministic wins shrink variance** — a real fix both raises the mean and
  tightens sd, a useful signature to distinguish signal from noise.
- Next: fold cold-fact extraction into the ingest proper (write to Knowledge
  Vault + retrieval key, per LongMemEval key-expansion), and resolve conflicting
  literals by recency.

### Repro
`evals/build_coldfacts.py <user>` → `coldfacts_<user>.json`; run the answerer with
`MIRIX_COLDFACT=1`. Negatives are gated `MIRIX_ENABLE_CONSOLIDATE` / `MIRIX_PERSONA`.

## 4c. Making the answerer actually USE the graph — still no gain

A standing caveat on every "the graph doesn't move QA" result was that the answerer
never really consumed the graph: its output was injected as a prompt context blob
that the model could skip, while the answerer's own `search_memory` (flat pgvector)
was what it actually read. That excuse is now removed.

Two changes:
1. `V7Retriever.retrieve_rows()` — the retrieval pipeline split so it can return
   **structured rows** (anchors, episodic, semantic) instead of only a rendered blob;
   `retrieve()` is now a thin formatting wrapper over it.
2. The eval answerer merges those rows into its **own `search_memory` results** —
   the exact delivery that made cold-facts work (+4.4), deduped against flat hits.

Wiring that up surfaced a silent bug worth recording: the answerer is a separate
process from the server and **never initialised the neo4j client**, so
`get_neo4j_driver()` returned `None` and the retriever returned empty on every call,
without an error or a log line. The graph had been contributing literally nothing
through this path. After initialising it, the graph does return real memories
(e.g. "vintage camera collection" → 6 memories flat search missed).

**A/B on one restored store, one variable, 3 runs each:**

| | runs | mean | sd |
|---|---|---:|---:|
| control (graph off) | 38 / 45 / 38 | 40.3 | 3.3 |
| graph merged into search results | 39 / 40 / 40 | **39.7** | 0.5 |

**−0.7 — no gain.** Per question it is a wash: 2 won (Q33, Q39), 2 lost (Q3, Q26).
(The control's 45 is an outlier; that arm's sd is 3.3.)

The mechanism behind the null result was measured directly: the graph's memories are
**largely already in the flat results**. On three probe queries the graph returned
15–16 memories and, after dedup, contributed 6 / 0 / 0 new ones. The graph is not
adding recall that flat vector search lacks.

So the conclusion survives its strongest test: it is not that the graph was wired up
wrong — once genuinely wired, it still does not help on this benchmark. Kept behind
`MIRIX_GRAPH_SEARCH` (default off).
