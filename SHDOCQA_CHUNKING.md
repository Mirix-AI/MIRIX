# SHDocQA on v7.1 — 4096 tokens vs 4096 characters

**Status: one arm complete, one arm blocked. The token arm scores 79/100. The char arm was
stopped by OpenAI rate limiting at 21 of 245 chunks and has no score.**

---

## The question

`evals/_chunking.py` records the character-based budget as a bug:

> Why this matters: char-based 4096 (the previous policy) emitted ~4× more chunks than
> official because 4096 chars ≈ 1024 tokens. That made MIRIX's retrieval look worse than
> apples-to-apples because each semantic unit was scattered across multiple memories.

That is a plausible claim and it was never measured. The policy was changed to tokens and
the old numbers were abandoned. This runs both on the branch where the original SHDocQA
numbers came from.

The 4× is real and reproduces exactly. On the actual RULER document (985,698 characters):

```
MIRIX_CHUNK_UNIT=token   4096 tokens  ->   50 chunks
MIRIX_CHUNK_UNIT=char    4096 chars   ->  245 chunks      4.9x
```

---

## Result

| | token | char |
|---|---|---|
| **accuracy** | **79/100 = 0.790** | — (blocked) |
| chunks ingested | 50 / 50 | **21 / 245** |
| questions answered | 100 | 0 |
| ingest wall-clock | 5.70 h | 0.66 h (partial) |
| median per chunk | 414 s | 93 s |
| episodic rows | 1014 | 152 |
| semantic rows | 1052 | 149 |
| rows outside the prefix | 0 | 0 |

The char arm died on `LLMRateLimitError` from OpenAI, surfacing as a 500 from
`/memory/add_sync` after the client exhausted its retries.

**Correction.** That was first read as request density — five times the chunks over the same
wall-clock hitting a rate ceiling. It was not. A later attempt returned
`429 ... You have no credits remaining`, so the account had simply run out of budget partway
through. The distinction matters: a rate ceiling would have been fixable by pacing the
ingest, and it is not. **Neither reading says anything about chunking** — do not take this as
"char chunking fails".

One number is worth extracting from the partial run anyway: the char arm's median chunk took
**93 s against the token arm's 414 s**, a ratio of 4.5, almost exactly the chunk-count ratio.
Per-chunk cost scales with chunk size, so the total ingest cost of the two policies is
probably similar; what differs is how the same work is divided. Extrapolating, the char arm
would need roughly 6.3 h of ingest — comparable to the token arm's 5.70 h.

---

## What the token arm establishes

79/100, with every verification column clean:

```
chunks=50           ingest actually ran; not answering from an empty store
questions=100       nothing skipped
episodic=1014       memories were written
outside-prefix=0    nothing landed in a foreign namespace
judged=100          score distribution {correct: 79, wrong: 21}
```

Graph was live throughout: 9,409 `V7Anchor` and 2,065 `V7MemoryRef` nodes written, and
`Graph retrieve` fired 200 times across 100 questions. `V7MemoryRef` is the ref-node layer
that v7.12 removed, which confirms this really is the v7.1 architecture and not the mainline
code wearing an old version string.

Scoring is the **substring** judge that RULER-style short answers use, not an LLM judge, so
this number carries no judge noise — it is deterministic given the answers.

### It does not reproduce the historical 87/100

`shdoc_v71` scored 87/100 in July 2026. That run is not comparable to this one:

- its prompts still contained LoCoMo gold answers (decontaminated 2026-08-04)
- it passed `--max-chunks 0`, which in this runner means `chunks[:0]` — **keep zero chunks**,
  not "no cap" — so it ingested nothing and answered against whatever the store already held
- it had no storage prefix, so it shared a namespace with every other run of that era

79 is what v7.1 scores with clean prompts and a real ingest of the full document.

---

## Four bugs found on the way, all of which produce a plausible number while doing nothing

Recorded because each one exits 0 and each one has bitten this project before.

**1. `--max-chunks 0` deletes the ingest.** `chunks = chunks[: args.max_chunks]`. The
historical scripts (`run_v81_docqa.sh:42`) all pass `--max-chunks 0` intending "no cap". To
get no cap, omit the flag. This explains why `shdoc_v71`, `shdoc_v8`, `shdoc_v81` and
`mhdoc_nograph` all show `chunks=0` in their stored results.

**2. `longmem_eval.py` cannot chunk a RULER document.** Its `parse_sessions` expects
LongMemEval's "Chat Time" structure; on a RULER document it returns zero chunks and the run
then answers all 100 questions against an empty store, exiting 0. Use `ruler_eval.py`, whose
`parse_documents` handles it.

**3. `mab/ruler_eval.py` had no storage prefix.** It wrote every row and graph node under the
bare sample id. A six-hour run produced 964 rows outside its own namespace before this was
caught. Now ported from `main_eval.py`.

**4. A comment can break a shell line continuation.** This is mine, not the repo's:

```bash
MIRIX_EVAL_USER_PREFIX="$prefix" MIRIX_CHUNK_UNIT="$unit" \
# a comment here
$PY -u mab/ruler_eval.py ...        # <- runs with NEITHER variable set
```

Both the prefix and the chunk unit were silently dropped — the arm ran with default settings
under the wrong namespace and looked normal.

The verification column caught every one of these, but only **after** the run. The lesson for
the next person: assert the namespace and the chunk count within the first minute of ingest,
not at the end.

---

## Reproducing

```bash
git worktree add /tmp/v71 graph_v7.1_clean
cd /home/lj/MIRIX_eval && ./v71_shdoc_chunk_ab.sh          # both arms, sequential
```

The script creates a fresh Postgres database and clears a dedicated Neo4j namespace per arm,
starts the server at `MIRIX_GRAPH_VERSION=v7.1`, runs `mab/ruler_eval.py --limit 1
--run-llm`, judges, and prints the verification line. Launch it with `setsid` — a plain
`nohup ... &` followed by a `sleep` in the same shell invocation gets SIGTERM'd along with
the parent when the caller times out, which killed four launches here.

To finish the char arm, throttle it. The failure is request density, so either pace the
ingest (a sleep between chunks) or raise the account's rate limit. Roughly 6-9 h.

---

## What this does and does not answer

**Answered:** v7.1 with token chunking, clean prompts and a full ingest scores **79/100** on
SHDocQA. The 4.9× chunk-count difference between the two policies is real and reproducible.
Per-chunk ingest cost scales with chunk size, so the two policies likely cost similar total
ingest time.

**Not answered:** whether char chunking actually scores worse. The repo's claim that scattering
a semantic unit across several memories hurts retrieval remains **unmeasured**. One arm is not
a comparison.

**Not comparable to:** anything on the mainline. This branch predates the v7.12 n-ary frame
rewrite, six AutoDream variants and five retrieval policies, and it still writes `V7MemoryRef`
nodes. Use these numbers only against other runs on this branch.
