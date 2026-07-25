# Patch note: raw-chunk side channel for the MemoryAgentBench adapter

**Scope.** `samples/memoryagentbench/mirix_adapter.py` only. No changes to
MIRIX core, MAB, or any prompt template.

**Status.** Opt-in. Default behaviour (`mirix_preserve_raw_chunks` unset)
keeps the adapter on the pure MIRIX retrieval path.

## Background

MIRIX's `add` endpoint pushes every ingested message through the meta agent
and its six sub-agents, which **abstract** the input into structured memory
items: `{name, summary, details, tree_path}` for semantic, event summaries
for episodic, and so on. The original chunk text is not retained verbatim
anywhere in the database, and `retrieve_with_conversation` returns these
abstracted items, never the source string.

That is the right behaviour for a personal assistant: a few months in,
nobody wants to grep raw screen captures for "what was my Wi-Fi password" —
they want a deduplicated, summarised memory.

It is the wrong shape for some MemoryAgentBench sub-datasets. The most
extreme case is **Conflict_Resolution / FactConsolidation**: the adapter
ingests a numbered list of contradicting facts,

```
0.   Thomas Kyd was born in the city of London.
...
306. Thomas Kyd was born in the city of Leeds.
```

and the gold answer is the entry with the **largest serial number**
(`Leeds`, even though the world-knowledge answer is `London`). MIRIX's
`semantic_memory_agent` collapses both entries into one summary item and
will sometimes annotate it with its own world-knowledge belief
(`"Thomas Kyd was born in London; some sources incorrectly claim Leeds"`).
Both the serial number and the verbatim wording — the only signals the
benchmark scores — are gone by the time `retrieve_with_conversation`
serves a query.

This affects any MAB sub-dataset whose gold answer depends on token-exact
content the summarising agents will discard: serial numbers, verbatim
excerpts, exact label-to-class mappings.

## Change

When `preserve_raw_chunks` is on, the adapter additionally keeps the
un-templated chunk in a per-`user_id` Python list at ingest time:

```python
# inside _memorize, after the regular client.add(...) call
if self.preserve_raw_chunks:
    self._raw_chunks[user_id].append(message)
```

At query time it BM25-ranks those raw chunks against the question, picks
the top-k (default 5), and prepends them to the retrieved-memory block
that goes into the prompt:

```
--- Raw ingested chunks (verbatim, ordered by BM25 relevance) ---
<chunk N>

<chunk M>

--- MIRIX memory retrieval ---
<flattened items returned by retrieve_with_conversation>
```

The MAB query template — the prompt with the rules and the `{question}`
slot — is **unchanged**. `get_template(...)` still resolves to MAB's
`templates.py` verbatim. Only the contents that fill the
"retrieved memory" slot in the prompt are augmented.

## Configuration

`mirix_preserve_raw_chunks` in the agent YAML, or `--preserve-raw-chunks`
on `run_bench.py` / `run_ablation.py`:

| value          | effect                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| unset / `null` | **off** (default). No local raw cache, no BM25, no extra tokens. Pure MIRIX semantics.                  |
| `false`        | Same as off, but explicit.                                                                              |
| `true`         | Force on for every sub-dataset.                                                                         |
| `"auto"`       | Adapter decides per sub-dataset using `mirix_adapter._RAW_CHUNK_RECOMMENDED_SUBDATASETS`.               |

The recommended list currently turns the side channel on for sub-datasets
matching `factconsolidation*`, `ruler_qa*`, `eventqa*`, `icl_*`,
`recsys_*`. Adding a new benchmark to that list is the only change needed
to opt it in to `auto`. CLI override beats YAML; YAML beats `auto`.

`mirix_raw_chunk_topk` (default 5) controls how many raw chunks BM25
returns per query.

## Why this is a fair comparison

MAB's other agentic-memory backends already do the equivalent verbatim
storage:

- **letta** in `insert` mode (the configuration used for MAB's main
  results) calls `passage_manager.insert_passage(text=formatted_message)`
  directly. The full chunk goes into letta's archival memory verbatim;
  letta's own memory-agent loop is bypassed.
- **mem0** writes the templated message into its vector store, which
  tokenises the chunk verbatim before embedding.

MIRIX exposes no such verbatim-passthrough lane in the public HTTP API:
the only writer endpoint is `add`, and `add` unconditionally routes
through the abstracting meta agent. The side channel is the smallest
external work-around that puts MIRIX on the same footing as those
backends for verbatim-critical benchmarks. With it disabled, MIRIX is
benchmarked on its own native retrieval semantics.

## Cost

Empirical numbers from FactConsolidation `sh_6k` (100 questions,
gpt-4o-mini, top-5 raw chunks):

| mode  | EM   | F1    | input tokens / question |
| ----- | ---- | ----- | ----------------------- |
| off   |  14% | 16.8% | ~4,700                  |
| auto  |  71% | 71.9% | ~17,100                 |

Per-question wall time rises by less than a second on this dataset.

For larger contexts (`sh_64k`, `sh_262k`), the BM25 selection becomes the
operative knob — a context split into 64 chunks with `topk=5` still puts
~20k tokens into the prompt regardless of context size, but the
likelihood that the right chunks are in the top-5 falls. `topk` is the
lever there.

## What this is not

- It is not a change to MAB's prompt templates. `templates.py` is
  imported and used unchanged.
- It is not a change to MIRIX core. Nothing under `mirix/` is touched.
- It is not a backdoor that lets MIRIX "cheat" — letta and mem0 already
  store chunks verbatim in their stores, and the side channel is the
  adapter's only way to put MIRIX on parity with that.
- It is not on by default. A benchmark run with no flags measures pure
  MIRIX retrieval.
