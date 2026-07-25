# Extractor Direction D — LLM triple extraction (keep relations, per-chunk)

Design note capturing the validated direction for the graph extractor, to be built
as a version next round. Supersedes the v7.4 GLiNER extractor for the concept gap.

## Why: the v7.4 GLiNER concept-loss (corrected finding)

v7.4 (GLiNER) shipped as a ~200× build-time win with the User/Assistant noise hubs
gone, at net-neutral QA (32/60 vs v7 30/60). **But a per-question look revealed a real
regression it hides:** GLiNER, a span tagger, **cannot abstract**. v7's LightRAG (an
LLM) read *"socializing with colleagues while working from home"* and named the concept
**"Networking" / "Team Collaboration" / "Remote Team Connections"**; GLiNER can only tag
spans that literally appear, so those concept anchors vanish (verified: `networking`,
`team collaboration`, `coworkers`, `remote team connections` = 0 in the v7.4 graph; v7
had them). Type mix confirms it: GLiNER over-produces `object` (907, 37%) and under-
produces `concept` (294, 12%). On concept-heavy queries the v7.4 graph fails to match
and the retriever falls back to off-topic memories (the "staying connected with
colleagues" query retrieved *fitness* memories under v7.4, but the right social/
networking anchors under v7). The neutral aggregate QA masked this.

Aggressive GLiNER tuning (threshold 0.3, 14 concept-ish labels) does **not** fix it —
it only catches more literal spans (`social interactions`, `watercooler conversations`),
never the abstraction. This is a hard limit, not a config gap.

## Why LLM: what recent top-venue graph works actually use

From the entity-extraction research (see the workflow), **every** recent peer-reviewed
KG/graph-memory method uses **LLM extraction of triples**; GLiNER is the outlier (a
general NER tool, not a KG builder):

| method | extraction | granularity | output |
|---|---|---|---|
| HippoRAG 2 (ICML'25) | 1 LLM OpenIE call | per passage | (s, p, o) triples |
| AutoSchemaKG (ACL'26) | 3 LLM passes (E-E, E-V, V-V) | per chunk | triples + events |
| KGGen (NeurIPS'25) | 2 LLM calls (entities, then relations) | per chunk | JSON triples + clustering |
| KARMA (NeurIPS'25) | 9 LLM agents, ~20-30 calls | per document | triples + verify |
| EDC (EMNLP'24) | few-shot LLM open IE | per document | [S, R, O] triples |
| GLiNER (NAACL'24) | encoder forward pass | — | spans, **no relations, no abstraction** |

Takeaways: (1) LLM, because abstraction + relations are the point; (2) they extract
**triples** — the relation (predicate) is kept, whereas v7 **discards** LightRAG's
relations; (3) granularity is **per passage/chunk**, not per memory.

So MIRIX's real inefficiency was never "using an LLM" — it was running LLM extraction
**twice** (6-agent chunk→memories, then LightRAG memory→anchors) and **throwing the
relations away**. v7.4 GLiNER "fixed" cost by removing the LLM, losing abstraction with
it. Direction D fixes cost the SOTA-aligned way instead.

## D: the direction

**One LLM triple-extraction call per chunk; keep the relations; concepts abstracted.**

|  | v7 LightRAG | v7.4 GLiNER | **D** |
|---|---|---|---|
| LLM | ✅ | ❌ | ✅ |
| abstraction | ✅ | ❌ | ✅ |
| relations | extracted, **discarded** | ❌ | ✅ **kept** (anchor→anchor edges) |
| provenance | — | — | ✅ carried in the relation (User seeks…/Assistant suggests…) |
| granularity | per memory (many calls) | — | **per chunk** (few calls) |

GLiNER is demoted to an optional cheap named-entity supplement.

### Validated prompt (gpt-4.1-mini, temperature 0, JSON mode)

```
Extract knowledge-graph triples from the text: every meaningful (subject, relation, object) fact.
Subjects/objects are ENTITIES: named things (people/places/orgs/products), OR the underlying CONCEPT/THEME.
For concepts, output a CONCISE CANONICAL name (2-4 words, the general theme), NOT a copied phrase:
  "misses watercooler chats with colleagues while remote" -> concept "Remote Team Networking"
  "socializing with colleagues while working from home"   -> concept "Workplace Socializing"
Each entity: name + type from person, organization, location, event, concept, method, content, object, date.
Return JSON: {"triples":[{"s":{"name","type"},"r":"short relation","o":{"name","type"}}]}
```

Validated output on the real "colleagues" memory (the query GLiNER failed) — 6 triples,
2.9 s:

```
(User)      --seeks-->    (Workplace Socializing [concept])   ← abstraction GLiNER can't do
(User)      --enjoys-->   (Remote Work [concept])
(User)      --misses-->   (Workplace Socializing [concept])
(Assistant) --suggests--> (Online Communities [concept])
(Assistant) --suggests--> (Virtual Coffee Breaks [method])
(Assistant) --suggests--> (Alumni Networks [organization])
```

Clean canonical concepts (comparable to v7's Networking/Social Connections), relations
kept, and User/Assistant provenance sitting in the relation subject.

## Build plan (next round)

1. `mirix/services/triple_extractor.py` — the prompt above → `[Triple(s,r,o)]` with typed
   entities. Async, one call per input.
2. `process_memory`: add `relations=` (alongside existing `entities=`). Build
   **anchor→anchor `V7_RELATION` edges** (new — v7 has none). Entities still go through
   `_select_anchors` and link to memory refs as today.
3. Per-chunk batching driver: group a chunk's memories, one extraction call, map entities
   back to the memory refs they appear in.
4. Gate on a new `graph_version` (e.g. `v7.6`); add to both guard tuples.
5. Verify on the same 962 memories: do abstract concept anchors return (networking, team
   collaboration…)? relation edges present? per-question fix on the colleagues query?
   cost (calls/time) vs v7 LightRAG and v7.4 GLiNER? QA vs both.

### Roadmap impact

D becomes the extractor foundation and provides the relation edges that **v7.6 (PPR
retrieval)** needs — so D and "keep relations + PPR" merge into one line. Role/domain
scoping (v7.5) also gets provenance for free from D's relation subjects. v7.4 GLiNER
stays shipped-but-superseded (a cheap named-entity supplement if ever wanted).
