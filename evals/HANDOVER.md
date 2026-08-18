# MIRIX — evaluation handover and runbook

**Goal: LoCoMo 90.** We are at 89.0–89.6% (mean of same-config runs, not a single draw).
HyperMem reports 92.73 on the same 1540 questions with a byte-identical judge, so the real
gap is 48–57 questions.

- **§1-§3** — how to run things: environment, LoCoMo, MemoryAgentBench.
- **§4-§7** — where the errors are, what each version changed, what to build next, where the files are.

If you read only two things: **§2.3 (judging — adopt majority-of-3)** and **§2.4
(assertions before every arm)**. Most of the wasted runs in this project came from skipping
one of those.

---

# PART I — RUNBOOK

## 1. Environment

```
repo      /home/lj/code/MIRIX            branch graph_revision
evals     /home/lj/code/MIRIX/evals
harness   /home/lj/MIRIX_eval            scripts, venv, .env, saved dumps
python    /home/lj/MIRIX_eval/.venv/bin/python
psql      /home/lj/MIRIX_eval/pgenv/bin/{psql,dropdb,createdb,pg_dump}
scratch   /tmp/claude-*/scratchpad       logs
```

`/home/lj/MIRIX_eval/.env` is the single source of runtime config. The run scripts **rewrite
it with `sed`** before starting the server, so do not hand-edit it while a run is in flight.
Keys that matter:

```bash
MIRIX_PG_DB=mirix_locomo_clean_r1     # which Postgres store
MIRIX_GRAPH_VERSION=v7.23             # retrieval policy + extraction gates
MIRIX_ENABLE_GRAPH_MEMORY=true        # false = flat Postgres only
MIRIX_GRAPH_RERANK=1
MIRIX_NEO4J_URI / _USER / _PASSWORD
OPENAI_API_KEY
```

### Which graph version to run

**Use `v7.23` for both ingest and QA** unless you have a reason not to.

| | LoCoMo, 1540 | note |
|---|---|---|
| v7.23 | **1367** (`clean_v723_full`), 1361 / 1370 (`ab_graph_r1/r2`) | the store on disk is built at this version |
| v7.24 | 1362 QA-only on the same graph | net −5, but that hides **multi-hop .883 → .830**, i.e. 53 questions lost and 48 regained elsewhere |
| v7.25 | recorded null in-repo | |
| v7.26 | −18 over four paired comparisons | see §5 |
| v7.12 | indistinguishable from v7.23 on 411 questions | but 36% cheaper to ingest |

v7.24 is the trap: it looks like a 5-question regression and is really a large loss in the
one category where we are furthest behind HyperMem (93.62 vs our .887). Do not use it.

v7.12 is a defensible choice **only** if ingest hours are the binding constraint — four MAB
tracks across two configurations is a lot of ingest, and v7.23's coverage-repair pass costs
36% more wall-clock for a difference no one has been able to measure. The cost is that none
of the results already on disk would be comparable to your new ones.

Set per-run in the scripts, not in `.env`:

```bash
MIRIX_EVAL_USER_PREFIX=clean-r1__     # NOT MIRIX_USER_PREFIX — that name is silently ignored
MIRIX_DREAM_EVERY_N_CHUNKS=5
MIRIX_DREAM_MODE=experience
MIRIX_WINDOW_TURNS=0                  # >0 splits sessions into N-turn windows at ingest
MIRIX_EPISODE_DEDUP=0                 # >0 folds same-day near-identical episodic rows
MIRIX_GRAPH_UNION=0                   # 1 runs flat search alongside the graph (measured: -2)
```

Always `unset MIRIX_COLDFACT MIRIX_HYBRID_SEARCH MIRIX_SEARCH_LIMIT` at the top of a run
unless the arm is about them — they persist in the shell and silently change results.

### Starting the server

```bash
cd /home/lj/MIRIX_eval && ./run_server.sh          # port 8531, reads .env
```

**Do not hand-roll `uvicorn mirix.server.server:app`.** It fails with "Attribute app not
found", and if your wait loop only retries without asserting, the eval will answer every
question with no memory and still exit 0. That happened; see §2.4.

---

## 2. Running LoCoMo

### 2.1 Full ingest + QA (the standard arm)

`/home/lj/MIRIX_eval/locomo_clean.sh` is the reference. Copy it for a new arm and change
`DB`, `PREFIX`, `BK`, the log name and the output paths. It does, in order:

1. drop and recreate the Postgres DB, `CREATE EXTENSION vector`
2. delete every Neo4j node under `PREFIX` (only that prefix — the graph is shared)
3. start the server at the requested graph version
4. `main_eval.py --data data/locomo10.json --run-llm --mirix_config_path ./configs/0201c_v6.yaml --output_path <name>`
5. `organize_results.py --judge default results/locomo/<name>`
6. `pg_dump` to `saved/`

Cost: **~5.5h ingest + ~1h QA** for 10 conversations. The graph roughly quintuples ingest
(12.8 → 61.6 s/chunk).

```bash
cd /home/lj/MIRIX_eval && nohup timeout 43200 ./locomo_clean.sh > /dev/null 2>&1 &
tail -f /tmp/claude-*/scratchpad/locomo_clean.log
```

Smaller datasets for iteration: `data/locomo_pilot3.json` (3 conversations, 411 questions),
`data/locomo_c30.json` (1 conversation, 105 questions).

### 2.2 QA-only against an existing store — use this whenever possible

Any change that does not alter what is *written* can be tested without re-ingesting, which
removes the larger of the two noise sources and costs ~1h instead of 6.5h.

`/home/lj/MIRIX_eval/graph_ablation.sh` is the template — **copy this for new retrieval or
answering arms.** The mechanism is: seed the output directory from a completed run, keeping
its chunk markers and clearing its answers, so `main_eval.py` skips ingestion.

```python
# inside the arm script
src = pathlib.Path("results/locomo/clean_v723_full")
dst = pathlib.Path("results/locomo") / arm_name
for f in src.glob("conv-*.json"):
    d = json.loads(f.read_text())
    d["records"] = {}          # re-answer everything, keep the chunk markers
    (dst / f.name).write_text(json.dumps(d, ensure_ascii=False, indent=2))
```

Then the same `main_eval.py --run-llm` line as above. It will see the markers and only answer.

### 2.3 Judging

```bash
cd /home/lj/code/MIRIX/evals
python organize_results.py --judge default results/locomo/<name>
```

Scores live in `results/locomo/<name>/metrics.json` under `llm_judge_results`. **Compute
accuracy yourself from that list, excluding `category == "5"`** — `overall_metrics` is
sometimes empty and the run scripts print a blank when it is.

```python
J = [x for x in json.load(open(p))["llm_judge_results"] if str(x.get("category")) != "5"]
acc = sum(1 for x in J if x.get("score") == 1) / len(J)      # denominator 1540
```

#### Adopt HyperMem's judging protocol — LLM-as-judge ×3, majority

This is the single cheapest change on this page and it should be made before any close
comparison. **We grade each question once. HyperMem grades it three times and takes the
majority**, which is why their numbers carry less noise than ours at the same accuracy.

From their code (`EverMind-AI/HyperMem`, `hypermem/main/stage6_eval.py`):

```python
num_runs = 3                                                        # :344
grading_tasks = [locomo_grader(...) for _ in range(num_runs)]       # :287  three concurrent
judgments = await asyncio.gather(*grading_tasks)
...
true_count = sum(1 for v in judgments.values() if v)                # :481
is_correct = true_count >= (num_runs / 2)                           # i.e. at least 2 of 3
```

It is **not** "all three must agree" — it is a 2-of-3 majority. gpt-4o-mini at temperature 0
is not deterministic in practice, and majority voting sharpens: an item whose per-call
probability of CORRECT is p becomes `p**3 + 3*p**2*(1-p)`, which is higher than p above 0.5
and lower below it. Borderline answers stop flipping between runs.

Measured on our side: single-call grading leaves **~25 questions of judge noise on 1540**
(1.6%). Cost of removing it is roughly **$1 per full run** — negligible against 5.5h of
ingest, and smaller than most of the effects we have been trying to detect.

`/home/lj/MIRIX_eval/hypermem_judge.py` implements their protocol exactly — same prompt,
same model, same temperature, same 3-vote majority, same lenient label parsing — and
re-scores any stored run without re-answering anything:

```bash
python hypermem_judge.py --run clean_v723_full
```

Two things worth knowing about the shared prompt. Their accuracy prompt and ours are
**byte-identical**; re-scored under their full protocol `clean_v723_full` is 0.8805 against
their 0.9273, so **judging is not the gap**. And neither judge actually reasons, despite the
prompt asking it to: the text opens with "First, provide a short explanation of your
reasoning" and closes with "Just return the label CORRECT or WRONG in a json format", and
gpt-4o-mini obeys the closing instruction. Run side by side, both emit a bare
`{"label": "..."}`.

The canonical prompt is `evals/llm_judge.py`, sha256-pinned with `assert_canonical_prompt()`.
Import it; never paste a copy. Four files once carried their own, and one used U+2019
apostrophes where the others used ASCII — worth 3 questions.

### 2.4 Assertions — run before every arm

```bash
cd /home/lj/code/MIRIX/evals
python assert_store_sane.py --db mirix_locomo_clean_r1 --prefix clean-r1__ --graph
```

Checks: rows exist under the expected prefix, **nothing sits outside it**, the server
answers on 8531, Neo4j has anchors for this prefix. Three runs in this project produced a
plausible number while not running the experiment:

- Neo4j died mid-run → 1540 questions answered from an empty memory, judged, `rc=0`, 0.3461.
- A run exported `MIRIX_USER_PREFIX` instead of `MIRIX_EVAL_USER_PREFIX`, so every row landed
  under the bare `conv-43` namespace on top of an older run's 959 anchors → `rc=0`, 0.882.
- A hand-rolled uvicorn line failed to load the app; 18 questions were answered with no
  server at all.

Also **log something inside the code path under test** that proves it fired, and read it
back. `MIRIX_SEARCH_LIMIT` was once "measured" as having no effect when the branch never
executed at all.

### 2.5 Running a no-graph vs with-graph comparison

There are two different questions here and they need two different experiments. Decide which
one you are answering before you start.

| question | design | what it costs |
|---|---|---|
| **Is MIRIX better WITH the graph?** — a claim about the system | two independent ingests, graph off and graph on, each run twice | ~15h on LoCoMo |
| Why is it better, and by how much on the read side? | one store, QA-only, flip the flag | ~4h |

**For a reported V1 / V2 number you want the first one.** A system claim has to include the
write side: a real no-graph deployment never builds the graph at ingest and never runs
AutoDream. The same-store ablation deliberately holds those fixed, so it answers a narrower
question — a good one, but not the one a table headed "MIRIX-V1 / MIRIX-V2" is answering.

**A gap to fill first: LoCoMo has no clean V1 number.**

```
V2 system   clean_v723_full   1367/1540   graph ingest + graph QA          exists
V1 system   --                            no-graph ingest + no-graph QA    MISSING
ab_nograph_r1/r2  1310 / 1295             no-graph QA over a GRAPH-BUILT store — not V1
locomo_nograph_all  1275                  a real V1 system, but contaminated prompts — unusable
```

Producing it is the cheapest thing on this page: a no-graph full ingest is ~1.6h against
~5.5h with the graph, plus ~1h of QA. Copy `locomo_clean.sh`, set
`MIRIX_ENABLE_GRAPH_MEMORY=false`, give it a fresh `DB` and `PREFIX`, and run it. One arm is
already readable — the graph is worth roughly +63, and ingest drift is ~±10 on 1540, so the
effect clears the noise on a single pair. Run each side twice only if you need the error bar
rather than the sign.

Everything below is about the second design — the read-side ablation — which is what
`graph_ablation.sh` implements.

**The wrong way, which is the obvious way:** build one store with the graph off, build
another with it on, compare the two totals. Every published number in this project before
this week was produced like that, and none of them mean what they appear to.

Two separate ingests are two different stores. Identical code re-ingested moves the LoCoMo
score by about **5 questions on 411** — extraction is a chain of LLM calls, and one run wrote
419 semantic rows where another wrote 597. That drift sits on top of whatever the graph is
worth, and on 411 questions it is the same size. The contaminated-era "graph is worth +96"
came from this design and does not survive it.

**The right way: one store, QA-only, flip the flag.** Ingest once with the graph ON, so the
Neo4j side exists. Then answer the questions twice against those byte-identical Postgres
rows, once with `MIRIX_ENABLE_GRAPH_MEMORY=true` and once with `false`. The no-graph arm
falls back to flat Postgres search over the same content. Nothing is re-ingested, so ingest
drift is removed entirely and what remains is the retrieval path.

`/home/lj/MIRIX_eval/graph_ablation.sh` does exactly this — copy it. Its shape:

```bash
DB=mirix_locomo_clean_r1                       # ONE store, never rebuilt
run_arm () {
  local name="$1" graph="$2"
  # rewrite .env, restart the server at this setting
  sed -i "s/^MIRIX_ENABLE_GRAPH_MEMORY=.*/MIRIX_ENABLE_GRAPH_MEMORY=$graph/" $EVAL/.env
  cd $EVAL && nohup ./run_server.sh > "$SC/server_$name.log" 2>&1 &
  # ... health check, then ASSERT (see 2.4) ...
  # seed the output dir from a completed run so ingestion is skipped (see 2.2)
  # main_eval.py --run-llm ... --output_path "$name"
  # organize_results.py --judge default
  G=$(grep -c "graph-owned pass" "$SC/server_$name.log")
  say "$name acc=$ACC  graph-owned passes=$G  (want >0 iff graph=$graph)"
}
run_arm ab_nograph_r1 false
run_arm ab_graph_r1   true
run_arm ab_nograph_r2 false      # repeat each arm — see below
run_arm ab_graph_r2   true
```

Three things that script does which a hand-rolled version will miss:

1. **A verification column per arm.** `graph-owned passes` must be >0 for a graph arm and 0
   for a no-graph arm. Without it you cannot tell an arm that ran from an arm that silently
   fell back — and this project has three runs that returned a plausible number while doing
   neither.
2. **Each arm twice.** The spread between two identical arms is the noise floor for that
   comparison, measured here at 9–15 questions on 1540. Report the mean of two, not a single
   draw.
3. **Nothing is re-ingested.** The seeding trick in §2.2 is what makes the whole thing cost
   ~4h instead of ~26h.

Reference result, so you know what a working run looks like — four arms, one store, clean
prompts, v7.23:

```
no graph   r1 1310/1529   r2 1295/1529      mean 1302.5
graph      r1 1361/1529   r2 1370/1529      mean 1365.5     the graph is worth +63.0

single-hop 830  .872 -> .923  +43.0     multi-hop 282  .846 -> .887  +11.5
temporal   321  .857 -> .875   +6.0     open-dom   96  .682 -> .708   +2.5
```

Most of the gain is single-hop: it works as an index far more than as a reasoning structure.

**What this does and does not measure.** Holding the store constant isolates the graph as a
RETRIEVAL PATH. A genuine V1 system would also never write the graph during ingest and would
never run AutoDream, which operates on it — so strictly, +63 is the read-side value, not the
whole system difference.

That distinction turns out not to matter here, and it is worth knowing why. Ingesting the
same three conversations with the graph off writes 399 episodic and 427 semantic rows;
with it on, 389 and 419. That 2.5% gap is far inside the run-to-run variation of identical
code, which produced 419 against 597 semantic rows on two runs of the same version. So there
is no evidence that enabling the graph changes what lands in Postgres — AutoDream merges
anchors in Neo4j without rewriting the rows underneath. Read-side value and system value are
the same number as far as anything here can tell.

For MemoryAgentBench you have no choice anyway: those runners ingest and answer in one pass,
so V1 and V2 are two ingests and the drift comes with them. Run each configuration twice.

For MemoryAgentBench the same discipline applies but the seeding trick does not, because
those runners ingest and answer in one pass. There you must pay for two ingests — so run
each configuration **twice** and compare means, and record ingest wall-clock and store size
alongside the score (§3.1), since on LoCoMo the graph costs 4.8x ingest time for its +63.

---

## 3. Running MemoryAgentBench — four tracks

Server must be up on 8531 (§1) and `datasets` installed in the same Python. Every track
writes the same per-sample schema, so `organize_results.py` works on all of them unchanged.

| # | track | HF `metadata.source` | runner | judge |
|---|---|---|---|---|
| 1 | LongMemEval-S | `longmemeval_s*` | `mab/longmem_eval.py` | `--mab-judge` (LLM) |
| 2 | **SHDocQA** — single-hop, SQuAD-derived | `ruler_qa1_197K` | `mab/longmem_eval.py --source` | substring |
| 3 | **MHDocQA** — multi-hop, HotpotQA-derived | `ruler_qa2_421K` | `mab/longmem_eval.py --source` | substring |
| 4 | **DetectiveQA** — multiple-choice mystery | `detective_qa` | `mab/lru_eval.py --source` | exact |

Note tracks 2 and 3 go through **`longmem_eval.py --source`**, not `ruler_eval.py`. Both can
load RULER rows, but `longmem_eval.py` is what the existing `shdoc_*` / `mhdoc_*` results
were produced with (`/home/lj/MIRIX_eval/run_v81_docqa.sh:42`), so use it or the numbers are
not comparable to what is already on disk.

`mab/ruler_eval.py` and the `infbench_sum_eng_shots2` source of `lru_eval.py` exist and work,
but are **not** in this four-track set.

### Commands

```bash
cd /home/lj/code/MIRIX/evals
PY=/home/lj/MIRIX_eval/.venv/bin/python

# 1 — LongMemEval-S (wrapper also snapshots the store afterwards)
LIMIT=5 MAX_QS=60 ./mab/run_mab_longmem_eval.sh          # fast arm, 60 QA
LIMIT=5           ./mab/run_mab_longmem_eval.sh          # full, 300 QA

# 2 — SHDocQA
$PY -u mab/longmem_eval.py --limit 1 --max-chunks 0 --run-llm \
    --source ruler_qa1_197K --output_path shdoc_<tag> \
    --mirix_config_path ./configs/mab.yaml

# 3 — MHDocQA
$PY -u mab/longmem_eval.py --limit 1 --max-chunks 0 --run-llm \
    --source ruler_qa2_421K --output_path mhdoc_<tag> \
    --mirix_config_path ./configs/mab.yaml

# 4 — DetectiveQA
$PY -u mab/lru_eval.py --limit 10 --run-llm \
    --source detective_qa --output_path det_<tag> \
    --mirix_config_path ./configs/mab.yaml

# judging, all four
$PY organize_results.py --mab-judge results/longmem/<output_path>
```

`--limit` is rows/conversations to ingest, `--max-chunks 0` means no cap (the RULER rows are
197K–421K tokens each, so this is the expensive part), `--max-questions` caps QA per row.
Smoke-test any config change with `--limit 1 --max-questions 5` before committing to an arm.

Sizes of the existing runs, for calibration: SHDocQA and MHDocQA are **100 questions** each
over 2 samples; the LongMemEval-S fast arm is **60**. Budget ~6h for the LongMemEval-S 60-QA
arm and 7–8h with the graph enabled.

Three judges, **not interchangeable** — a substring-scored track and an LLM-judged track are
different scales and must not share a table row:

```
mab/llm_judge_mab.py          port of MemoryAgentBench longmem_qa_evaluate.py
mab/llm_judge_substring.py    exact needle match, for RULER-style short answers
mab/llm_judge_mab_summary.py  port of their summarization_evaluate.py (infbench_sum only)
```

Prior results already on disk, useful as controls:

```
results/longmem/  shdoc_nograph shdoc_v7 shdoc_v71 shdoc_v8 shdoc_v81 shdoc_hybrid
                  mhdoc_nograph mhdoc_v7 mhdoc_v71 mhdoc_v72 mhdoc_v8 mhdoc_v81
                  c_baseline_det c_pruned_det          (DetectiveQA)
                  c_baseline c_pruned ...              (LongMemEval-S)
```

### 3.1 The V1 / V2 comparison that is still owed

```
MIRIX-V1   MIRIX_ENABLE_GRAPH_MEMORY=false
MIRIX-V2   MIRIX_ENABLE_GRAPH_MEMORY=true, MIRIX_GRAPH_VERSION=v7.23,
           MIRIX_DREAM_EVERY_N_CHUNKS=5
```

Run all four tracks under both configurations, and report **three columns per cell**, not
one. These runners ingest and answer in one pass, so V1 and V2 are necessarily two separate
ingests — which is the right design for a system claim (§2.5), but it means ingest drift
rides along. **Run each configuration twice and compare means**, or the number carries an
uncertainty nobody can see. Only performance has been collected so far:

| | performance | memory size | latency |
|---|---|---|---|
| MIRIX-V1 | done | **missing** | **missing** |
| MIRIX-V2 | done | **missing** | **missing** |

Both missing columns are recoverable from what the runs already write — **no re-run needed**:

```bash
# memory size, per store
psql -d "$DB" -At -c "SELECT count(*), sum(length(coalesce(summary,'')||coalesce(details,'')))
                      FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE '$PREFIX%';"
# same for semantic_memory; and in Neo4j:
#   MATCH (n) WHERE n.user_id STARTS WITH $p RETURN labels(n)[0], count(*)
```

```python
# latency, per results/<bench>/<run>/conv-*.json
t = json.load(open(f))["timings"]
ingest_s, qa_s = sum(t["add_chunk"].values()), sum(t["answer"].values())
```

For reference, on LoCoMo the graph costs **4.8× ingest wall-clock** (12.8 → 61.6 s/chunk)
and roughly nothing at QA time, for +63 questions. That ratio belongs in the table — a
performance column alone hides it.

---

# PART II — WHAT IS ALREADY KNOWN

## 4. Where the errors are

`ab_graph_r1`, 1540 questions, 169 wrong, triaged with controls at every stage:

| bucket | n | share |
|---|---|---|
| **INGEST** — never written to the store | 60 | 36% |
| **GOLD** — not derivable from the source | 41 | 24% |
| **RETRIEVAL** — in the store, never retrieved | 38 | 22% |
| **ANSWERING** — evidence present, answer wrong | 30 | 18% |

Two ceilings: an **oracle retriever** (answer from the 15 store rows that best match each
question's own gold) rescues **67 of 169** — 102 errors are beyond any ranking work; and
**gold defects cap the benchmark near 97.3%**.

### Why facts are not written

The extraction prompt asks for a summary that is "concise and informative", and the extractor
keeps the *point* of a turn rather than its *content*:

```
"Hey Jo, guess what I did? Dyed my hair last week"
   stored: "Nate dyed his hair purple last week"        (0 rows in the store contain "Jo")

"I'm currently reading 'The Lean Startup' and hoping it'll give me tips for my biz"
   stored: "Jon is wrapping up a business plan..."      (0 rows contain the title)

Caroline recommends "Becoming Nicole"; Melanie reads it
   stored: "Caroline inspired by book 'Becoming Nicole'"
           "Melanie read inspirational book last year"  (the link is broken)
```

Aggravated by `episodic_memory_merge` **overwriting** the previous summary — every merge is
another chance to drop a detail — and by one chunk carrying several people's events, which
crosses their attributions.

## 5. v7.12 → v7.24, version by version

| version | what changed | status |
|---|---|---|
| v7.13 | role canonicalisation | gated to v7.13 **only**; off in every later version |
| v7.14–v7.18 | five AutoDream variants | superseded (v7.15 measured worst, 131/152 on conv-26) |
| v7.19 | bounded dirty-frontier AutoDream | **live** — used by v7.19 through v7.24 |
| v7.20 | precision policy: exact-match boost, low-degree anchors beat hubs, person hubs penalised, lexical pass after vector recall | accumulated |
| v7.21 | second lane: exact anchor → predicate-filtered facts → cited memories | accumulated |
| v7.22 | predicate normaliser; rank by relation + entities + object + temporal; separate quotas | accumulated |
| v7.23 | adaptive evidence policy **and** a coverage-repair second extraction pass | **live** |
| v7.24 | role/time/state/citation policy | live for QA-only; **worse than v7.23 on multi-hop** (.830 vs .883) |
| v7.25 | one fused ordering key | recorded null in-repo |
| v7.26 | candidate admission | **−18, do not use** |

Three things to know:

1. **v7.19's dream never touches Facts** — `graph_reconsolidator_v719.py:377` reports
   `"online_fact_cleanup": "disabled"` unconditionally, final cycle included. Merge
   candidates must both be semantic-backed and share a type from
   `{person, organization, location, object, concept}`. It processed ~4989 anchors with only
   364 pairs rejected, so budget is not the constraint — **`_PAIR_COS = 0.93` is**, and it
   hides 71% of lexically redundant pairs from ever becoming candidates.
2. **v7.23 bundles two changes.** Its docstring says retrieval policy, but it also enables a
   second LLM extraction pass (`frame_extractor.py:487`). Its +7 mixes an ingest change and a
   retrieval change pulling in opposite directions, and it costs 36% more ingest time.
3. **Every retrieval version keeps candidate generation inside Neo4j by design.** When the
   graph owns episodic/semantic, their flat searches are never scheduled
   (`rest_api.py:3399`). Running both in union was tested: **−2**.

## 6. The one experiment worth running next

**Topic-grouped, multi-episode extraction at ingest** — HyperMem's actual mechanism and the
only one on this page with an external existence proof at 92.73.

`hypermem/main/stage2_hypergraph_extraction.py:561` gathers **all** episodes of a topic —
topics span weeks and are explicitly non-contiguous — and makes **one** extraction call over
the lot. `hypermem/prompts/fact_prompts.py:51-71` then demands two passes: per-episode facts,
then *combined* facts spanning several episodes, each carrying multiple episode ids. **The
cross-episode join is precomputed at write time.** No read-side policy reconstructs it, which
matches our graph being worth only +11.5 on multi-hop against their 93.62.

Validate it **without QA first**: three conversations, two arms, a deterministic store metric
(does a fact linking two sessions exist at all?). `/home/lj/MIRIX_eval/window_recall.py` is a
worked example of a judge-free store metric. Only pay for a full re-ingest if that clears.

## 7. File map

```
/home/lj/MIRIX_eval/
  (all of these are now also in evals/harness/ — see its README)
  locomo_clean.sh          full LoCoMo ingest + QA — copy for a new ingest arm
  graph_ablation.sh        4-arm QA-only template — copy for a new retrieval/answer arm
  hypermem_judge.py        HyperMem's grader, majority-of-3, re-scores any stored run
  final_triage.py          the four-bucket error triage, with positive controls
  final_triage.json        per-question verdicts
  window_recall.py         judge-free store metric (worked example)
  coldfact_gate.py         the offline gate that killed the cold-fact lane
  HANDOVER.md              this file

/home/lj/code/MIRIX/evals/
  main_eval.py             ingest + QA driver; MIRIX_WINDOW_TURNS lives in iter_sessions
  assert_store_sane.py     pre-flight — run before every arm
  llm_judge.py             canonical judge prompt, sha256-pinned
  organize_results.py      judging; --judge default (LoCoMo) / --mab-judge (MAB)
  mab/                     MemoryAgentBench runners
  data/                    locomo10.json, locomo_pilot3.json (3 conv), locomo_c30.json (1)

results under evals/results/longmem/  (MemoryAgentBench):
  c_baseline / c_pruned ...        LongMemEval-S
  shdoc_nograph v7 v71 v8 v81      SHDocQA   (ruler_qa1_197K)
  mhdoc_nograph v7 v71 v72 v8 v81  MHDocQA   (ruler_qa2_421K)
  c_baseline_det / c_pruned_det    DetectiveQA

results under evals/results/locomo/:

  USE THESE — clean prompts (2026-08-04 onwards), and the ab_* arms share ONE store
  clean_v723_full          1367/1540   cleanest full run; the store every QA-only arm seeds from
  ab_graph_r1 / _r2        1361 / 1370 same-store graph arms
  ab_nograph_r1 / _r2      1310 / 1295 same-store no-graph arms  -> the graph is worth +63
  ab_v726_r1 / _r2         1347 / 1348 candidate admission, -18, kept as a negative result

  DO NOT QUOTE — produced before the prompts were decontaminated on 2026-08-04
  locomo_nograph_all       1275/1540 (0.8279)   the old "no graph" number
  v712_full_..._noocr_r1   1355/1540 (0.8799)   what it used to be compared against
  v721_full_... v722_... v723_... v724_...      every pre-August full run

  The 0.8279 figure is wrong in three ways at once and the errors do not cancel. Its prompts
  contained LoCoMo's gold answers, and a no-graph arm depends on the store's contents more
  than a graph arm does, so the contamination does not affect both sides equally. It was
  built on its own store (saved/locomo_nograph_all/mirix_locomo_ng.dump), and so was the run
  it was compared with, so the ~80-question gap between them also contains ingest drift,
  independently measured at ~5 questions per 411. Re-measured properly the no-graph arm
  scores 1302.5, twenty-four questions HIGHER than 0.8279, and the graph's contribution is
  +63 rather than +80 or the +96 quoted elsewhere. The old number understates the flat
  system and overstates the graph simultaneously.
```
