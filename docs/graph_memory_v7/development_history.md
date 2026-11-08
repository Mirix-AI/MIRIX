# Development history — how the graph memory got here

The research arc from the v7 baseline to the current system (hypergraph +
reranked retrieval + lossless interleaved consolidation), including the failed
branches and the measurements that killed them. Written so a reader can follow
*why* each design exists, not just what it is. Companion records:
`version_ledger.md` (per-version table), `extraction_recovery.md` (LongMemEval
results), `locomo_dream_ablation.md` (LoCoMo ablation), `archive/` (superseded
design-era docs).

## Phase 1 — baseline and retrieval variants (v7 – v7.3)

v7's founding principle: **content stays in flat PG; graph nodes must earn
their place by creating a retrieval path**. LightRAG extraction, anchors with
surface dedup, memory refs.

- v7.1 reranked anchor-collected candidates by query text-cosine — **SH-Doc +4,
  the first graph change to beat flat on held memories**. (This rerank later
  went silently dead for every newer version; see Phase 6.)
- v7.2 per-anchor coverage round-robin — neutral.
- v7.3 proposition-based ingest — **rejected**: collapsed the edge structure
  (only DESCRIBED_BY survived, singletons ↑77%).

## Phase 2 — the extractor line (v7.4 – v7.11)

- v7.4 GLiNER: ~200× faster, hub-free — but a span tagger cannot abstract;
  concept anchors vanished. Kept only as a named-entity supplement.
- v7.6 "direction D" LLM triples: concept anchors back **plus relation edges**
  the graph never had. The honest accounting: 2.6× faster than LightRAG (an
  earlier 34× claim compared concurrent vs sequential runs and was retracted).
- v7.7 Personalized PageRank over those relations: retrieval visibly richer, QA
  flat — first hint that the answerer, not the graph, was the bottleneck.
- v7.8 registry-guided entity resolution at extract time: 18% anchor reduction
  where cosine-only managed 2%.
- v7.9 role-split retrieval context — rejected (34→30).
- v7.10 **hypergraph**: each triple reified as a V7Fact node joining subject
  anchor, object anchor and source memory, with role + timestamp. Facts become
  citable, dedupable, governable — the structural foundation everything later
  stands on.
- v7.11 `consolidate` answerer tool for counting questions — rejected (36→30/31).

## Phase 3 — the measurement crisis

Running the *same* config four times gave 36/30/33/30: **mean 32.2, sd 2.5, a
third of questions flipping run to run**. Every prior version-to-version QA
delta was inside this band. Two rules came out of it and governed everything
after: (1) no conclusion from a single run; (2) a real fix both raises the mean
and tightens the sd (deterministic wins), which is how signal is told from
noise. Per-question forensics then split the stably-wrong questions into a
taxonomy: ~61% answerer-side, ~17% write-time extraction loss, ~13% temporal —
and ~0% pure retrieval misses.

## Phase 4 — the write-time turn

Answerer-side interventions (counting tool, temporal prompt, persona store,
cold-fact *injection*) all landed at or below baseline. What worked was
recovering what ingest had dropped:

- **Cold-fact recovery +4.4** (32.2→36.7, sd 0.5): a second pass over raw
  source extracts literal-bearing facts; merged into `search_memory` results
  **as ordinary evidence**. Framing was load-bearing — the same facts injected
  as "ground truth" netted zero because they overrode correct answers.
- **Rebuilt clean graph +4.0** (→40.7): the gains clustered exactly where graph
  structure should help (temporal ordering, multi-session aggregation).

Lesson that held from here on: **every real gain came from what gets written,
never from how the answerer is told to read.**

## Phase 5 — the consolidation saga

auto_dream (upstream: an LLM agent that merges memories) was made to run at
scale (proportional batching for a 142k-token overflow; a stale version guard
that routed v7.4+ to a dead legacy graph builder; invisible logging), then
measured: **no-dream 40.3 vs with-dream 32.7 — consolidation cost −7.7.**

The diagnosis went through three stages, each forced by a user challenge:

1. "Merging shouldn't lose anything" → strengthened preserve-facts prompt.
   Facts survived in the store, QA didn't recover — the harm was also
   *retrieval degradation* (blended embeddings can't find what they contain).
2. **graph_only mode**: skip the flat merge entirely, refine only the graph.
   Byte-identical PG proven with before/after hashes → QA break-even *by
   construction*, anchors −27%. First dream mode that satisfied "至少持平,
   結構更精煉".
3. On LoCoMo the flat merge was caught **literally deleting** single-occurrence
   facts (sunflower/figurine/pride-festival grep to 0 rows post-dream; the
   answerer's "there is no record" was true). Root cause: "merge" was an LLM
   rewrite + hard delete, with removal as the prompt's first goal — a soft
   constraint fighting the primary objective.

The fix made the constraint mechanical: the **union-coverage gate**. Before any
replace/update deletes, every deterministic specific of the old rows (numbers,
dates, month/weekday words, word-numbers, multi-word proper names) must appear
in the replacement text, else the call is rejected pre-delete and the error
steers the model to rewrite. Verified live: 29 lossy merges rejected, temporal
accuracy recovered from .838 to .919 (the exact no-dream level), consolidation
still happening.

## Phase 6 — structural hygiene, found by measuring invariants

Measuring the graph's degree distribution and post-dream state kept exposing
silent defects, each with the same signature — an invariant that should hold,
didn't:

- A **"Users" anchor at degree 1016** (7.4× the biggest real hub, 10.5% of all
  facts): the per-extractor noise blocklists missed plurals and LightRAG had no
  filter at all → one canonical-key gate at `_select_anchors`, plus a
  retroactive purge pass in maintenance.
- The **v7.1 rerank had been dead** for every version after v7.1 (`== "v7.1"`
  guard), and the retriever dispatcher stranded v7.4–v7.10 on a dead v5
  pipeline. Same stale-exact-match bug class in four sites; all now
  `startswith`. Paired A/B after revival: +2~3 on LoCoMo.
- **Consolidation grew the graph** (+27% facts) instead of shrinking it —
  merged-away memories left zombie facts with no citations, and DETACH-deleting
  refs shredded the temporal chain (52 edges/100 refs). Two maintenance passes
  restored the invariant (post-fix: dreamed graph converges to the no-dream
  graph's size; chain n−1).
- **Live ingest hooks never passed `role`** — every organically built graph had
  silently lost user/assistant attribution; only rebuilt graphs had it, because
  the rebuild script passed it itself. Found when the offline-rebuild shortcut
  was banned (see below).

## Phase 7 — LoCoMo cross-validation and the ablation

A user-enforced methodology rule paid off twice: **evals must build PG and
graph together through the real ingest hooks** (entity resolution is
path-dependent; the paper describes the online system). Enforcing it exposed
the role-loss bug above, and the fresh-ingest discipline made the six-arm
ablation clean (each arm its own end-to-end build):

no-graph .8816 / graph .8882 / **graph+rerank .9079** / ungated dream .8618 /
ungated+rerank .8750 / **gated dream+rerank 134-137-135 ≈ no-dream band**.

Three durable findings: the reranker is real (+2~3 paired, second benchmark
after SH-Doc); the graph's per-category value is multi-hop (.846→.923 in every
graph arm); and the graph is **insurance** — worth ~0 when the flat store is
intact, ~+24/152 when the flat store has been consolidated (its facts were
extracted before the merge). Consolidation damage is question-type-dependent:
LongMemEval (cold specifics) −7.7/60 vs LoCoMo (salient gist) −2~5/152, and the
gate closes most of the latter.

## Phase 8 — the regime lesson

A LongMemEval rerun accidentally used a different chunking path (session-split
4096-char, 534 chunks) and scored 38/60 where the canonical baseline was ~30 —
because fine-grained ingest natively captured the cold facts (500 Mbps, 7
shirts, $420: 0 rows in the canonical store, 2–7 rows in the fine-grained one).
Two lessons: **chunking granularity moved QA more than most retrieval work**
(worth a controlled study), and benchmark numbers are only comparable within a
regime — the canonical one is `evals/mab/longmem_eval.py`, 114 × 4096-token
sentence-aware chunks, timestamps attached.

## Phase 9 — current direction: incremental semantic consolidation (v2)

The accepted redesign of auto_dream, synthesizing everything measured:

- **every 5 chunks** (affordable because incremental);
- new memories probe the existing graph by **embedding + shared-anchor
  neighborhood** instead of whole-store batching (fixes batch-blindness: the
  same topic weeks apart never met inside a batch);
- **episodic is immutable** — the event log and temporal chain are never
  rewritten (temporal damage removed by construction);
- consolidation output is **just a semantic memory** (no new node type):
  additive insert (or in-place update of an existing consolidation row),
  `source="auto_dream"`, provenance via `V7_SUPPORTED_BY(reason=consolidation)`
  edges to every raw source ref — original rows never modified;
- the union-coverage gate is reused as the constructor invariant (the
  synthesized text must cover all source specifics), and conflicts are resolved
  as recorded history ("was X (5/06) → now Y (6/20)").

## Meta-lessons

1. **Noise discipline first.** Without the 32.2±2.5 floor, half the rejected
   ideas above would have "worked".
2. **Mechanical guarantees beat instructions.** The preserve-facts prompt
   reduced loss; the coverage gate ended it. Same pattern as the noise-anchor
   gate vs per-extractor blocklists.
3. **Write-time beats read-time.** Cold facts, rebuilt graph, chunking
   granularity — the levers were all on the write side.
4. **Measure invariants, not just accuracy.** Zombie facts, chain fragmentation,
   role loss and the dead rerank were all invisible in QA numbers and obvious
   in structure checks.
5. **The user's design instincts repeatedly beat the implementation's
   defaults** — "merging shouldn't lose", "build PG and graph together",
   "consolidate only the knowledge layer" each exposed a real defect or a
   better architecture; the record is in Phases 5–9.
