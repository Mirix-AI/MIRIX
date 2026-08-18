# MIRIX evaluation — every result, and the settings that produced it

Companion to `HANDOVER.md`, which says how to run things. This one says what came out.

Read §0 first. Half the numbers in this repository were produced under conditions that make
them uncomparable, and the table that tells you which is which is more useful than any single
score.

---

## 0. Which numbers can be quoted

Three cutoffs decide whether a stored result means anything.

| cutoff | date | effect |
|---|---|---|
| **prompt decontamination** | 2026-08-04 (`36d8d40`) | before this, the extraction prompts and the answerer's few-shot examples contained LoCoMo gold answers — "Becoming Nicole", "clarinet and violin", the Melanie/Caroline pair. Every earlier number compares a system against answers it was shown. |
| **storage prefix** | added per-runner at different times | without `MIRIX_EVAL_USER_PREFIX`, a run writes to the bare conversation id and shares a namespace with every other run of its era. The shared Neo4j still holds unprefixed `conv-42`/`conv-43`/`conv-48` from this. |
| **same-store discipline** | 2026-08-08 | before this, "graph vs no graph" was two separate ingests, and identical-code re-ingest is independently measured at ±5 questions per 411. |

**Quotable:** `clean_v723_full`, `clean_v724_qaonly`, `ab_graph_r1/r2`, `ab_nograph_r1/r2`,
`ab_v726_r1/r2`, `p3*` (August), and the v7.1-branch SHDocQA run.

**Not quotable:** everything before 2026-08-04 — `locomo_nograph_all` (0.8279),
`v712_full_*`, `v721_full_*`, `v722_*`, `v723_full_*`, `v724_full_*`, every `lc26_*`,
`v710ctl`, `v71x_*`, and the whole historical MemoryAgentBench batch.

---

## 1. Settings

### 1.1 Environment (`/home/lj/MIRIX_eval/.env`)

The run scripts rewrite this file with `sed` before starting the server, so it is not stable
between arms — read it from the script that produced a result, not from disk.

```bash
MIRIX_PG_HOST=localhost           MIRIX_PG_PORT=5432        MIRIX_PG_USER=mirix
MIRIX_PG_DB=<per arm>             MIRIX_PG_BIN=/home/lj/MIRIX_eval/pgenv/bin
MIRIX_NEO4J_URI=bolt://localhost:7687
MIRIX_NEO4J_USER=neo4j            MIRIX_NEO4J_DATABASE=neo4j
MIRIX_NEO4J_VECTOR_DIM=1536
MIRIX_ENABLE_GRAPH_MEMORY=true    # false = flat Postgres only
MIRIX_GRAPH_VERSION=v7.23         # retrieval policy AND extraction gates
MIRIX_GRAPH_RERANK=1              # 0 = unranked traversal-order truncation
```

### 1.2 Per-run, set in the script

```bash
MIRIX_EVAL_USER_PREFIX=clean-r1__   # NOT MIRIX_USER_PREFIX — that spelling is ignored silently
MIRIX_DREAM_EVERY_N_CHUNKS=5
MIRIX_DREAM_MODE=experience
MIRIX_WINDOW_TURNS=0                # >0 = N-turn ingest windows, 1 turn overlap
MIRIX_EPISODE_DEDUP=0               # >0 = fold same-day near-identical episodic rows
MIRIX_WRITE_DEDUP=0                 # semantic equivalent
MIRIX_GRAPH_UNION=0                 # 1 = run flat search alongside the graph
MIRIX_CHUNK_UNIT=token              # v7.1 branch only; char restores the pre-fix budget
MIRIX_COLDFACT / MIRIX_HYBRID_SEARCH / MIRIX_SEARCH_LIMIT   # unset these unless testing them
```

### 1.3 Models

| role | model | note |
|---|---|---|
| ingest agents | gpt-4.1-mini | via `evals/configs/0201c_v6.yaml` |
| answerer | gpt-4.1-mini | temperature 0, seed 42 |
| judge (LoCoMo) | gpt-4o-mini, temperature 0 | prompt sha256-pinned in `evals/llm_judge.py` |
| judge (RULER) | substring exact match | deterministic, no judge noise |
| embeddings | **text-embedding-ada-002** | the configs say `text-embedding-3-small`, but `OpenAIEmbedding` is constructed without a model argument so llama_index's default wins. Every store in this repo is ada-002. |

### 1.4 Chunking

| benchmark | unit |
|---|---|
| LoCoMo | one session per `add_chunk`, no token budget |
| MemoryAgentBench | 4096 **tokens** of the gpt-4o-mini tiktoken encoding, sentence-aware (NLTK punkt), no mid-sentence split |

`--max-chunks 0` means `chunks[:0]` — **keep zero chunks**, not "no cap". To get no cap, omit
the flag.

---

## 2. LoCoMo — the headline results

1540 non-adversarial questions (category 5 excluded), clean prompts, LLM judge.

### 2.1 The graph ablation — four arms over ONE store

The only clean answer to "what is the graph worth". Nothing re-ingested; byte-identical
Postgres rows on every arm; only `MIRIX_ENABLE_GRAPH_MEMORY` differs.

```
no graph   r1 1320/1540 (.8571)   r2 1305/1540 (.8474)     mean 1312.5
graph      r1 1371/1540 (.8903)   r2 1380/1540 (.8961)     mean 1375.5
                                                    the graph is worth +63.0 (+4.1 points)
```

| category | n | no graph | graph | Δ |
|---|---|---|---|---|
| single-hop | 830 | .872 | .923 | **+43.0 questions** |
| multi-hop | 282 | .846 | .887 | +11.5 |
| temporal | 321 | .857 | .875 | +6.0 |
| open-domain | 96 | .682 | .708 | +2.5 |

**It is a good index, not a reasoning structure.** Two thirds of the gain is single-hop.
Cost: the graph roughly quintuples ingest, 12.8 → 61.6 s/chunk.

### 2.2 Every quotable LoCoMo run

| run | score | multi | temporal | open | single | ingest | QA |
|---|---|---|---|---|---|---|---|
| `ab_graph_r2` | 1380/1540 .8961 | .883 | .879 | .708 | .929 | — | 1.17h |
| `ab_graph_r1` | 1371/1540 .8903 | .890 | .872 | .708 | .918 | 5.54h | 1.06h |
| `clean_v723_full` | 1367/1540 .8877 | .883 | .872 | .698 | .917 | 5.54h | 1.04h |
| `clean_v724_qaonly` | 1362/1540 .8844 | **.830** | .875 | .719 | .925 | — | 1.01h |
| `ab_v726_r2` | 1358/1540 .8818 | .848 | .879 | .698 | .916 | — | 1.02h |
| `ab_v726_r1` | 1357/1540 .8812 | .858 | .869 | .677 | .917 | — | 1.04h |
| `clean_v724_gpt5mini` | 1351/1540 .8773 | .833 | .869 | .740 | .911 | — | **5.14h** |
| `ab_nograph_r1` | 1320/1540 .8571 | .848 | .860 | .698 | .878 | — | 1.01h |
| `ab_nograph_r2` | 1305/1540 .8474 | .844 | .854 | .667 | .867 | — | 1.04h |

**Our number is the mean of same-config runs, ~89.0-89.6%, not a single draw.** Against
HyperMem's 92.73 that is a gap of 48-57 questions, not the 4.7 points a single run suggests.

Two things to notice in that table. `v7.24` (`clean_v724_qaonly`) looks like a 5-question
regression against v7.23 and is really **multi-hop .883 → .830** — 53 questions lost, 48
regained elsewhere — in the one category where the gap to HyperMem is largest. And
`gpt5mini` costs five times the QA wall-clock to score 16 questions lower.

### 2.3 Three-conversation arms (411 questions)

| run | score | what it tested |
|---|---|---|
| `p3fix_v724_qaonly` | 369/411 .8978 | three ingest fixes |
| `p3r2_v724_qaonly` | 368/411 .8954 | **same code, re-ingested — the control** |
| `p3r2_v723_full` | 367/411 .8929 | |
| `p3fix_v723_full` | 366/411 .8905 | |
| `p3union` | 363/411 .8832 | flat search alongside the graph |
| `p3ng_full` | 356/411 .8662 | no graph, own ingest |
| `p3v712_full` | 203/233 .8712 | v7.12, incomplete (conv-43 aborted) |

The first two lines are the most important in this document. Three ingest fixes scored +6
against a stored baseline and **+1 against a same-code control**. The baseline was one draw.

---

## 3. Noise floors — read before comparing anything

### 3.1 QA re-runs: 9-15 questions on 1540, and it is retrieval

Two identical QA-only runs over one store, paired question by question:

| | count |
|---|---|
| retrieval returned byte-identical rows | 1174 / 1529 (77%) |
| ...of those, score flipped | 18 (**1.5%**) |
| retrieval returned different rows | 355 / 1529 (23%) |
| ...of those, score flipped | 19 (**5.4%**) |
| discordant total | 37, net +9 |

The answerer runs at temperature 0 with a fixed seed and the judge is near-deterministic, so
the drift is Neo4j's vector search plus an untied `ORDER BY sim DESC`. **Use McNemar on the
discordant pairs; for a retrieval change, restrict to questions whose context actually
differed.** Twelve graph versions moved the benchmark 16 questions in total — not
unmeasurable, measured with the wrong statistic.

### 3.2 Ingest re-runs: ±5 questions on 411

Identical code, fresh ingest: 419 vs 597 semantic rows between two runs of the same version.
Never score an ingest change against a stored baseline.

### 3.3 Judge: ~25 questions on 1540

Single-call grading. HyperMem grades three times and takes a 2-of-3 majority
(`stage6_eval.py:344/287/481`); adopting that costs about $1 per full run.

### 3.4 Not every benchmark has this problem

SHDocQA re-run against the same store gave **79/100 twice, with an identical per-question
score distribution** — not merely the same total. The substring judge is deterministic, and
RULER answers are short exact needles, so a shift in retrieval order rarely changes whether
the needle appears in the answer at all.

| benchmark | judge | same-config re-run spread |
|---|---|---|
| SHDocQA / MHDocQA | substring | **0 questions** |
| LoCoMo | LLM | 9-15 on 1540 |
| LongMemEval-S (60 q) | LLM | 7 on 60 — see §6.4 |

**Budget accordingly:** the RULER tracks can be compared from single runs; LoCoMo and
LongMemEval-S need two runs per arm. That halves the cost of four of the eight V1/V2 cells.

---

## 4. Where the errors are

`ab_graph_r1`, 169 wrong, triaged with a positive control at every stage.

| bucket | n | share |
|---|---|---|
| **INGEST** — never written to the store | 60 | 36% |
| **GOLD** — not derivable from the source | 41 | 24% |
| **RETRIEVAL** — in the store, never retrieved | 38 | 22% |
| **ANSWERING** — evidence present, answer wrong | 30 | 18% |

Two ceilings. An **oracle retriever** — answer from the 15 store rows that best match each
question's own gold — rescues **67 of 169**, so 102 errors are beyond any ranking work. And
**gold defects cap the benchmark near 97.3%**.

### Why facts are not written

The prompt asks for a summary that is "concise and informative", and the extractor keeps the
*point* of a turn rather than its *content*:

```
"Hey Jo, guess what I did? Dyed my hair last week"
   stored: "Nate dyed his hair purple last week"        0 rows in the store contain "Jo"

"I'm reading 'The Lean Startup' hoping it'll give me tips for my biz"
   stored: "Jon is wrapping up a business plan..."      0 rows contain the title

Caroline recommends "Becoming Nicole"; Melanie reads it
   stored: "Caroline inspired by book 'Becoming Nicole'"
           "Melanie read inspirational book last year"  the link is broken
```

---

## 5. Interventions tried, with numbers

### 5.1 Answering side — eleven attempts, all net-negative or null

Replayed against stored contexts, so retrieval variance is excluded. Target = 110 questions
currently wrong with the gold present in their own context. Control = questions currently
right. **The control set is twelve times larger, so any unconditional instruction loses.**

| arm | target | control | note |
|---|---|---|---|
| control | 14/110 | 200/200 | replay alone rescues 14 |
| discriminate | 20/110 | 196/200 | name the failure mode |
| cite | 23/110 | 177/200 | forced verbatim citation — best target, worst control |
| twostage | 18/110 | 165/200 | select-then-answer |
| selfconsist | 20/110 | 197/200 | 3 samples, majority |
| **verifyretry** | **14/110** | 198/200 | **target moved by exactly zero** |
| fielded / grouped / dated | 25-26/110 | 177-181/200 | WHO\|WHEN\|WHAT tables |

`verifyretry` is the informative one: asked whether its own wrong answer was supported by the
evidence, the model says yes. **It cannot detect its own error.**

The only intervention that ever worked was `countfirst` (0/20 → 11/20 on counting questions)
— because it fired only on questions starting "how many". It had a detectable trigger.

Also tried: **gpt-5-mini as the answerer, −11 on the full run** at five times the QA cost.

### 5.2 Retrieval side

| change | result |
|---|---|
| flat + graph union (`MIRIX_GRAPH_UNION=1`) | **−2** on 411 |
| v7.26 candidate admission (64 anchors, 120-row window) | **−18** mean over four paired comparisons, two at p<0.05 |
| anchor entity resolution (simulated, 1829 merges) | connects the evidence for **1 of 31** failing multi-hop questions |
| verbatim cold-fact lane | gated out offline — ranking works (gold turn rank 1 for 24%, top-3 for 37%) but no threshold fires selectively: 0.86 reaches 24% of targets and 78% of controls |

v7.26's split is the mechanism: it **rescues 31-38 and breaks 51-55**. Widening admits the
right row *and* more plausible neighbours, and the second effect is larger. Four independent
measurements now say the same thing — more evidence in front of the answerer makes it worse.

### 5.3 Ingest side

| change | result |
|---|---|
| predicate-shaped facts + write dedup + speaker attribution | **+1** against a same-code control |
| 6-turn ingest windows (conv-30) | distinctive-token recall 73.7% → 73.7%, QA 72→73/81, **store 76% larger**. It *did* recover the traced case — rows containing "Lean Startup" went 0 → 3 — but one conversation yields only ~19 measurable tokens. |

---

## 6. MemoryAgentBench

### 6.1 The four tracks

| track | source | runner | judge |
|---|---|---|---|
| LongMemEval-S | `longmemeval_s*` | `mab/longmem_eval.py` | LLM |
| SHDocQA | `ruler_qa1_197K` | `mab/ruler_eval.py` | substring |
| MHDocQA | `ruler_qa2_421K` | `mab/ruler_eval.py` | substring |
| DetectiveQA | `detective_qa` | `mab/lru_eval.py` | exact |

**Substring-judged and LLM-judged tracks are different scales. Never put them in one table
row.**

### 6.2 The one clean MAB number

SHDocQA on the `graph_v7.1_clean` branch, decontaminated prompts, full ingest of the
985,698-character document:

```
v7.1, 4096 tokens        79/100 = 0.790
    chunks 50/50, questions 100, episodic 1014, semantic 1052, outside-prefix 0
    ingest 5.70h, median 414 s/chunk, QA 3.6 min
    graph live: 9,409 V7Anchor + 2,065 V7MemoryRef, Graph retrieve fired 200 times

v7.1, 4096 chars         BLOCKED at 21/245 chunks by OpenAI rate limiting
    median 93 s/chunk — 4.5x cheaper per chunk against a 4.9x chunk-count ratio,
    so per-chunk cost tracks chunk size and total ingest cost is probably similar
```

`V7MemoryRef` is the ref-node layer v7.12 removed, which confirms this is genuinely the v7.1
architecture. Whether char chunking scores worse **remains unmeasured** — one arm is not a
comparison.

The QA was re-run against the same store and reproduced **79/100 exactly**, per question.
Its 21 errors are all single-entity lookups clustered on proper nouns — "What was the Norman
religion?", "Who was Emma's brother?" — the same failure as LoCoMo's largest error bucket,
appearing on encyclopedic text instead of conversation.

The char arm's stop was **spent API credits** (`429 ... You have no credits remaining`), not
a rate ceiling. The first reading implied a fix — pace the ingest — that would not have
worked.

### 6.3 The historical batch — none of it is quotable

| run | score | chunks ingested |
|---|---|---|
| `shdoc_v71` | 87/100 | **0** |
| `shdoc_v8` | 85/100 | **0** |
| `shdoc_hybrid` | 84/100 | **0** |
| `shdoc_v7` | 83/100 | 50 |
| `shdoc_nograph` | 83/100 | **0** |
| `shdoc_v81` | 82/100 | **0** |
| `mhdoc_v7 / v71 / v72` | 84/100 | 106 / 0 / 0 |
| `mhdoc_v8 / v81` | 83/100 | **0** |
| `mhdoc_nograph` | 77/100 | **0** |

Every run showing `chunks=0` passed `--max-chunks 0`, which deletes the ingest. Whether those
scores mean anything depends on whether the script's "restore v7 graph" step succeeded, and
the output is identical either way.

**The 87/100 is not evidence that v7.1 is the best version.** The retriever gated its rerank
on `== "v7.1"`, so v7.3, v7.10, v8 and v8.1 all fell through to unranked truncation. That
table measures rerank versus no rerank. The equality was later widened to v7.1-and-above,
which is why v7.23 reranks today, and `MIRIX_GRAPH_RERANK` now toggles it directly.

### 6.4 LongMemEval-S at 60 questions cannot resolve anything

About ninety runs, scores from 29/60 to 45/60. Same-config repeats:

```
gsoff_1/2/3        38, 45, 38     spread 7 questions = 11.7 points
gson_1/2/3         39, 40, 40
coldfact_1/2/3     30, 32, 33
withdream_1/2/3    35, 32, 31
noise_base_1/2/3   30, 33, 30
```

**Use the 300-question version.** Nearly every comparison in that batch sits inside its own
noise, including "graph vs no graph" (`lm_v7_nograph` 30 against `lm_v8` 37).

### 6.5 Still owed

| | performance | memory size | latency |
|---|---|---|---|
| MIRIX-V1 (no graph) | — | — | — |
| MIRIX-V2 (graph + AutoDream) | — | — | — |

All eight cells across the four tracks need re-running with clean prompts. Memory size and
latency are recoverable from what the runs already write (`sum(length(summary||details))`,
`timings.add_chunk` / `timings.answer`) — no re-run needed once the performance column
exists.

---

## 7. Competitor check

**HyperMem 92.73 is real and comparable.** Their per-category figures uniquely determine
their numerators against our category totals, and they sum to the headline exactly:
841→808 (96.08), 282→264 (93.62), 321→288 (89.72), 96→68 (70.83); 1428/1540 = 92.7273.
Category 5 excluded at all three sites. Their accuracy prompt is byte-identical to ours —
re-scored under their full protocol `clean_v723_full` is 0.8805 against their 0.9273, so
**judging is not the gap**.

Their mechanism: `stage2_hypergraph_extraction.py:561` gathers **all** episodes of a topic —
topics span weeks and are explicitly non-contiguous — and makes **one** extraction call over
the lot, with a second pass producing facts that span several episodes, each carrying
multiple episode ids. **The cross-episode join is precomputed at write time.** Our graph is
worth +11.5 on multi-hop against their .9362; no read-side policy reconstructs that.

**Two other 9x% claims do not reconcile.** Synthius-Mem's 94.4 includes 442 adversarial
questions scored at 99.55% in its denominator — 92.7 without them, and on 11% fewer
non-adversarial questions than the standard split. Every system in HINDSIGHT's table reports
an overall 1.7-9.9 points above what its own per-category numbers imply.

Per category against HINDSIGHT, we are ahead on three of four: single-hop .920 vs .862,
multi-hop .887 vs .708, temporal .875 vs .838. Their lead is entirely open-domain
(.951 vs our .708) — the most judge-sensitive category, scored with a judge nobody else uses.

---

## 8. Silent failures — every one exited 0 with a plausible number

| what happened | what it produced |
|---|---|
| Neo4j died mid-run | 1540 questions answered from an empty memory, judged, `rc=0`, **0.3461** |
| `MIRIX_USER_PREFIX` instead of `MIRIX_EVAL_USER_PREFIX` | every row under bare `conv-43`, on top of an older run's 959 anchors, `rc=0`, **0.882** |
| hand-rolled `uvicorn mirix.server.server:app` | fails to load the app; wait loop only retried; **18 questions answered with no server** |
| `--max-chunks 0` | `chunks[:0]` — ingest deleted, questions answered against whatever the store held |
| `longmem_eval.py` on a RULER document | `parse_sessions` returns zero chunks; 100 questions answered against an empty store |
| `mab/ruler_eval.py` had no storage prefix | a six-hour run wrote **964 rows outside its own namespace** |
| a comment between `VAR=x \` and the command | drops every variable on that line; the arm ran with defaults under the wrong namespace |
| `MIRIX_SEARCH_LIMIT` treated as a cap | it is a floor; the "budget cut changed nothing" experiment had never cut anything |

`evals/assert_store_sane.py` catches the first three. The rest are why every arm should print
a verification line — rows under the expected prefix, rows outside it, chunks ingested, and
whether the code path under test actually fired — **within the first minute, not at the end**.
One of these burned six hours before the end-of-run check caught it.

---

## 9. What is still unmeasured, ranked

1. **The write-time cross-episode join** — HyperMem's mechanism, the only one with an
   external existence proof at 92.73, and untried here.
2. **Open-domain** — 96 questions at .708, contributing 17% of all errors from 6% of the
   benchmark, and the most stable category across runs. Nothing anyone has tried touches it.
3. **char vs token chunking** — one arm blocked.
4. **MIRIX-V1 vs V2 across the four MAB tracks** — eight cells, none clean.
5. **`MIRIX_EPISODE_DEDUP`** — written, unit-checked, never run.

Realistic ceiling for everything currently on the list: **90.2-91.4%**. The three read-side
buckets cannot close a 3.1-3.7 point gap, because retrieval's own oracle only reaches 67 of
169 errors and the answerer bucket is 30.
