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

## 5. Takeaways

- **The lever is write-time, not the answerer.** Recovering summarized-away facts
  is the only intervention that beat the noise band.
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
