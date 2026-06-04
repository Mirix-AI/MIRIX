# MAB eval suite — design + change log

## Overview

This change adds a MAB (MemoryAgentBench) eval suite covering four benchmarks — LongMemEval-S, RULER QA1/QA2, and LRU (infbench + detective) — and lands two small robustness fixes plus a Redis modules-missing latch in MIRIX core. The eval suite is commit `ca9bb36` ("Add MAB eval suite: LongMemEval-S, RULER QA1/QA2, LRU infbench/detective"); the MIRIX core patches are commit `b27fe93` ("MIRIX: Redis modules-missing latch + small robustness fixes"). Both commits sit on branch `mab_runner`, currently 2 commits ahead of `origin/main`. A backup branch `mab_runner_pre_squash_260604-1402` preserves the pre-squash 14-commit history as a local-only safety net.

## The MAB eval suite

Commit `ca9bb36` ("Add MAB eval suite: LongMemEval-S, RULER QA1/QA2, LRU infbench/detective") adds an end-to-end harness for four MemoryAgentBench (MAB) benchmarks that all run against MIRIX via the remote client. Every runner shares the same shape — `MirixMemorySystem.add_chunk` ingest → `wrap_user_prompt` → `TaskAgent.answer` → judge → snapshot — and writes the same per-sample JSON schema so `organize_results.py` works on every output unchanged. Sample IDs are namespaced per-source so multiple subsets can share one Postgres DB.

### Runners (one per dataset)

#### [`evals/longmem_eval.py`](/Users/weichiehhuang/MIRIX_eval/evals/longmem_eval.py) — 433 lines

- **Dataset / HF source:** `ai-hyz/MemoryAgentBench`, split `Accurate_Retrieval`, filtered to rows whose `metadata.source` equals `--source` (default constant `DEFAULT_LONGMEM_SOURCE = "longmemeval_s*"`, an exact match that mirrors the official MAB judge CLI `--dataset longmemeval_s*`). One HF row = one ~1.6M-char "long context" plus a parallel list of questions / answers / `question_types` / `question_ids` ([longmem_eval.py:40-90](/Users/weichiehhuang/MIRIX_eval/evals/longmem_eval.py#L40-L90)).
- **Per-sample loop ([longmem_eval.py:290-429](/Users/weichiehhuang/MIRIX_eval/evals/longmem_eval.py#L290-L429)):**
  1. `sample_id = f"longmem_s_{idx}"` (line 74). Construct `TaskAgent` and `MirixMemorySystem`, sharing the same `MirixClient` so both halves of the eval hit the same in-memory connection pool.
  2. `parse_sessions(context)` ([longmem_eval.py:96-178](/Users/weichiehhuang/MIRIX_eval/evals/longmem_eval.py#L96-L178)) — `ast.literal_eval`s the HF `context` field (alternating `"Chat Time: 2022/11/17 (Thu) 12:04"` strings and per-session message lists), parses each header with a regex into ISO 8601 (`"2022-11-17T12:04:00"`), prepends the header to the body, and pushes each session through `chunk_text_into_sentences` so each sub-chunk inherits the session's `occurred_at`. Returns `[{occurred_at, text}, ...]`.
  3. Reset server-side token counter via `POST /debug/token_stats/reset`, then ingest each chunk: `memory_system.add_chunk(chunk["text"], raw_input=..., occurred_at=...)`. Per-chunk wall time is recorded into `timings.add_chunk[idx]`, and the chunk's response goes into `responses[idx] = {type, chunk_index, question_index: None, occurred_at, response}`. Idempotent — re-runs skip indices already in `responses` (line 326).
  4. Snapshot ingest token stats into `token_stats.build_raw` / `build_sum`, then call `measure_memory_size(sample_id)` and stash under `memory_stats`.
  5. For each question, call `memory_system.wrap_user_prompt(question)` (records `timings.wrap_user_prompt[qidx]`), then `task_agent.answer(input_messages, user_id=sample_id)` (records `timings.answer[qidx]`). The record (`records[qidx_key]`) stores `sample_id, question_index, question_id, question, expected_answer, evidence:None, category, input_messages, predicted_answer, messages, usage, usage_total`. The `_abs` suffix on `question_id` routes to the MAB abstention prompt later.
  6. After QA, dump full PG memories with `dump_memories(sample_id)` to `<sample_id>_memories.json` and write `token_stats.query_raw` / `query_sum` (post-QA snapshot minus build totals).
- **CLI flags:** `--limit`, `--max-chunks`, `--max-questions`, `--run-llm` (default `True`), `--output_path` (default `Path("longmem_run")`; relative paths resolve under `evals/results/longmem/`; absolute paths warn), `--source` (default `"longmemeval_s*"`), `--mirix_config_path`.
- **Env vars:** `MIRIX_CLIENT_ID` (default `"mirix-eval-client"`), `MIRIX_ORG_ID` (default `"mirix-eval-org"`).
- **Output schema (`<sample_id>.json`):** `{sample_id, timings: {add_chunk, wrap_user_prompt, answer}, responses: {idx → ...}, records: {qidx → ...}, memory_stats, token_stats: {build_raw, build_sum, query_raw, query_sum}}`. Sidecar `<sample_id>_memories.json` is the full PG dump from `dump_memories`.

#### [`evals/ruler_eval.py`](/Users/weichiehhuang/MIRIX_eval/evals/ruler_eval.py) — 388 lines

- **Dataset / HF source:** `ai-hyz/MemoryAgentBench`, split `Accurate_Retrieval`, `metadata.source` in `{"ruler_qa1_197K"` (SHDOCQA, single-hop, SQuAD-derived; default), `"ruler_qa2_421K"` (MHDOCQA, multi-hop, HotpotQA-derived)`}` ([ruler_eval.py:43](/Users/weichiehhuang/MIRIX_eval/evals/ruler_eval.py#L43)). One row = one ~1-2M-char needle-in-haystack context of concatenated `Document N:` blocks + 100 short QA pairs whose answers are lists of acceptable substrings (judge must be `substring` — no LLM call needed).
- **Sample-id namespacing ([ruler_eval.py:76-90](/Users/weichiehhuang/MIRIX_eval/evals/ruler_eval.py#L76-L90)):** `ruler_qa1_197K → "shdocqa"`, `ruler_qa2_421K → "mhdocqa"`. `sample_id = f"{source_tag}_ruler_qa_{idx}"` so both subsets can share the same Postgres `user_id` namespace without collisions in `episodic_memory.user_id`.
- **Per-sample loop:** identical to `longmem_eval` ([ruler_eval.py:245-384](/Users/weichiehhuang/MIRIX_eval/evals/ruler_eval.py#L245-L384)) except the chunker is `parse_documents(context)` ([ruler_eval.py:116-133](/Users/weichiehhuang/MIRIX_eval/evals/ruler_eval.py#L116-L133)). That returns `[{occurred_at: None, text}, ...]` — RULER has no timestamps, so `occurred_at` is always `None`. The function no longer pre-splits on `Document N:` markers (the official chunker doesn't either); sentences span document boundaries but match leaderboard chunk shape.
- **CLI flags:** `--limit`, `--max-chunks`, `--max-questions`, `--run-llm`, `--output_path` (default `Path("ruler_run")`; relative under `evals/results/ruler/`), `--source` (default `"ruler_qa1_197K"`), `--mirix_config_path`.
- **`question_types`/`question_ids`:** empty lists ([ruler_eval.py:97-99](/Users/weichiehhuang/MIRIX_eval/evals/ruler_eval.py#L97-L99)) — RULER has no categorization, abstention is moot.
- **Output schema:** same as longmem_eval.

#### [`evals/lru_eval.py`](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py) — 404 lines

- **Dataset / HF source:** `ai-hyz/MemoryAgentBench`, split `Long_Range_Understanding`, `metadata.source` in `{"infbench_sum_eng_shots2"` (100 rows, novel summarization, default), `"detective_qa"` (10 rows, multiple-choice mystery QA)`}` ([lru_eval.py:52](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py#L52)).
- **Sample-id namespacing:** `infbench_sum_eng_shots2 → "infbench"`, `detective_qa → "detective"` ([lru_eval.py:82-95](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py#L82-L95)).
- **Per-sample loop ([lru_eval.py:250-400](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py#L250-L400)):** same as `ruler_eval`. Chunker is `parse_context` ([lru_eval.py:125-138](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py#L125-L138)) — also `chunk_text_into_sentences`, `occurred_at` always `None`.
- **Summarization-judge plumbing ([lru_eval.py:99-108](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py#L99-L108) and [357-379](/Users/weichiehhuang/MIRIX_eval/evals/lru_eval.py#L357-L379)):** for `infbench_sum_eng_shots2` each row carries `metadata.keypoints` and `metadata.qa_pair_ids`, which the runner propagates into each record as `keypoints` and `qa_pair_id` so `llm_judge_mab_summary` can score without a re-fetch from HF. For `detective_qa` these are empty (substring judge ignores them).
- **CLI flags:** `--limit`, `--max-chunks`, `--max-questions`, `--run-llm`, `--output_path` (default `Path("lru_run")`; relative under `evals/results/lru/`), `--source` (default `"infbench_sum_eng_shots2"`), `--mirix_config_path`.
- **Output schema:** same as longmem_eval, plus `records[qidx_key].keypoints` and `qa_pair_id` for the summarization judge.

#### [`evals/main_eval.py`](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py) — 315 lines (LoCoMo runner, modified in `ca9bb36`)

- **Dataset:** LoCoMo, loaded from a local JSON path (`--data`, default `data/locomo10.json`) via `load_locomo` ([main_eval.py:24-32](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py#L24-L32)). Each item has `sample_id`, `conversation` (with `session_N` lists and `session_N_date_time`), and `qa` list.
- **Per-sample loop ([main_eval.py:178-311](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py#L178-L311)):**
  1. `iter_sessions(conversation)` walks `session_<N>` keys; `format_session_chunk` prepends the LoCoMo `instructions` block and timestamp header, then dumps each turn as `speaker: text`.
  2. `memory_system.add_chunk(chunk, raw_input=chunk)` (no `occurred_at` — LoCoMo embeds the timestamp in the prompt instead).
  3. For each `qa` entry, `wrap_user_prompt(question)` → `task_agent.answer(...)` → record with `sample_id, question_index, question, expected_answer, evidence, category, input_messages, predicted_answer, messages, usage, usage_total`.
  4. `dump_memories(sample_id)` → `<sample_id>_memories.json` ([main_eval.py:300-311](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py#L300-L311)).
- **CLI flags:** `--data`, `--limit`, `--max-questions`, `--run-llm`, `--output_path` (default `Path("results")`), `--mirix_config_path`.
- **Output schema:** identical to the others, minus `token_stats` / `memory_stats` (those were added only on the new runners).

### Shared helpers

#### [`evals/_chunking.py`](/Users/weichiehhuang/MIRIX_eval/evals/_chunking.py) — 98 lines

Port of `utils/eval_other_utils.chunk_text_into_sentences` from the official MAB repo. Used by all three new runners so RULER, LRU and LongMemEval ingest the same shape the leaderboard runs use.

- **Atom:** NLTK `sent_tokenize` (auto-downloads `punkt_tab` then `punkt`, both `quiet=True`, via `_ensure_nltk()` at import time — [_chunking.py:37-64](/Users/weichiehhuang/MIRIX_eval/evals/_chunking.py#L37-L64)).
- **Budget:** `DEFAULT_CHUNK_TOKENS = 4096`, measured with `tiktoken.encoding_for_model("gpt-4o-mini")` (`DEFAULT_TOKEN_MODEL`). Encoder is cached at import as `_ENCODING`.
- **Public function:** `chunk_text_into_sentences(text: str, chunk_size: int = DEFAULT_CHUNK_TOKENS) -> List[str]` ([_chunking.py:68-98](/Users/weichiehhuang/MIRIX_eval/evals/_chunking.py#L68-L98)). Greedy: pile sentences into the current buffer until the next one would push past `chunk_size`, flush with `" ".join(buf)`, start a new buffer with the overflowing sentence. No mid-sentence split — a single sentence over the budget still ships in its own chunk (matches official). Empty input returns `[]`.
- **Why this matters:** char-based 4096 emitted ~4× more chunks than the official chunker (~1024 tokens), spreading each semantic unit across multiple memories and tanking retrieval. The module docstring spells this out.

#### [`evals/_eval_db.py`](/Users/weichiehhuang/MIRIX_eval/evals/_eval_db.py) — 183 lines

Direct-PG helpers. Both call `psql` via `subprocess.run` with connection params from `MIRIX_PG_{HOST,USER,DB,PASSWORD}` env vars (defaults `localhost / mirix / mirix / mirix`) and `psql` binary from `shutil.which("psql")` falling back to `/usr/local/opt/postgresql@17/bin/psql` ([_eval_db.py:25-32](/Users/weichiehhuang/MIRIX_eval/evals/_eval_db.py#L25-L32)).

- **`measure_memory_size(sample_id) -> Dict`** ([_eval_db.py:74-139](/Users/weichiehhuang/MIRIX_eval/evals/_eval_db.py#L74-L139)). For each of `episodic_memory` and `semantic_memory`, scalar-queries `count(*)` and `sum(length(coalesce(summary,'')||coalesce(details,'')))` (semantic also folds in `name`), keyed on `user_id`. Returns `{unit: "chars+tokens", flat: {table → {rows, chars}}, flat_total_chars, graph: {...}, graph_total_chars, total_chars, total_tokens}` where `total_tokens = total_chars // 4`. Also queries Neo4j (`MATCH (n) WHERE n.user_id=$uid` then `MATCH (a)-[r]->(b) WHERE a.user_id=$uid`) for node/edge counts and char totals, then reports `total_chars = max(flat_total, graph_total)` so flat-PG and graph backends are on the same yardstick.
- **`dump_memories(sample_id) -> Dict`** ([_eval_db.py:142-183](/Users/weichiehhuang/MIRIX_eval/evals/_eval_db.py#L142-L183)). Streams all rows for `user_id = sample_id` from `episodic_memory` and `semantic_memory` using `_pg_rows` (which uses `-F "\x01" -R "\x02"` so embedded newlines in summaries don't break row parsing). Returns `{user_id, memories: {episodic: {total_count, items: [...]}, semantic: {total_count, items: [...]}}}`.
- **Why bypassing `/memory/components` matters:** that endpoint caps each memory type at 50 by default and 200 even with an explicit `limit`. On conversations with thousands of memory items, the resulting `_memories.json` token count would silently under-report. Going straight to PG sidesteps the cap.

#### [`evals/llm_judge_substring.py`](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_substring.py) — 78 lines

Deterministic, no-LLM judge. Mirrors official `utils/eval_other_utils.substring_exact_match`.

- **`normalize_answer(text)`** ([llm_judge_substring.py:23-36](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_substring.py#L23-L36)): lowercase → strip `string.punctuation` → drop `a|an|the` at word boundaries → collapse whitespace.
- **`_accepted_answers(expected)`** ([llm_judge_substring.py:39-53](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_substring.py#L39-L53)): normalises gold answers from list, semicolon-joined string (`flatten_answer` output), or `None` into a flat `list[str]`.
- **`evaluate_substring_judge(predicted, expected) -> int`** ([llm_judge_substring.py:56-78](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_substring.py#L56-L78)): returns `1` iff `normalize_answer(predicted)` contains `normalize_answer(any accepted answer)` as a substring; `0` otherwise. Used by RULER QA1/QA2 and `detective_qa`.

#### [`evals/llm_judge_mab.py`](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py) — 144 lines

Port of `llm_based_eval/longmem_qa_evaluate.py`.

- **Metric model:** `DEFAULT_METRIC_MODEL = "gpt-4o"` ([llm_judge_mab.py:34](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py#L34)) — *not* `gpt-4o-mini` like the generic LoCoMo judge. `temperature=0`, `max_tokens=10`, `n=1`.
- **Five task-category prompts** ([llm_judge_mab.py:37-85](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py#L37-L85)):
  - `_SINGLE_AND_MULTI` — used for `single-session-user`, `single-session-assistant`, `multi-session`.
  - `_TEMPORAL` — for `temporal-reasoning`; verbatim "do not penalize off-by-one errors" clause.
  - `_KNOWLEDGE_UPDATE` — for `knowledge-update`; accepts responses that contain previous + updated info.
  - `_PREFERENCE` — for `single-session-preference`; "does not need to reflect all the points in the rubric".
- **Plus `_ABSTENTION`** ([llm_judge_mab.py:77-85](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py#L77-L85)) — routed to whenever `abstention=True`, regardless of category. Reframes the question as "unanswerable".
- **`SUPPORTED_TASKS`** ([llm_judge_mab.py:88-97](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py#L88-L97)) is the frozenset of the five known categories.
- **`get_anscheck_prompt(task, question, answer, response, abstention=False)`** ([llm_judge_mab.py:100-118](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py#L100-L118)): if `abstention`, return `_ABSTENTION`; else route by `task`; else `NotImplementedError`.
- **`evaluate_mab_judge(...) -> int`** ([llm_judge_mab.py:121-144](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab.py#L121-L144)): scoring rule is the verbatim official binary rule — `1 if "yes" in completion.lower() else 0`. Not JSON-parsed CORRECT/WRONG.

#### [`evals/llm_judge_mab_summary.py`](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py) — 294 lines

Port of `llm_based_eval/summarization_evaluate.py`.

- **Metric model:** `DEFAULT_METRIC_MODEL = "gpt-4o-2024-05-13"` — pinned, *not* the `gpt-4o` alias (alias drifts, diverges from leaderboard). `DEFAULT_TEMPERATURE = 0.1`, `DEFAULT_MAX_TOKENS = 4096` ([llm_judge_mab_summary.py:51-53](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L51-L53)).
- **Three calls per record** ([llm_judge_mab_summary.py:233-294](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L233-L294)):
  1. **Fluency** (`_FLUENCY_PROMPT_BOOK`, [llm_judge_mab_summary.py:56-72](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L56-L72)) — binary 0/1 on coherence/non-repetitiveness. Returns `{"fluency": 0 or 1}`.
  2. **Recall** (`_RECALL_PROMPT_BOOK`, [llm_judge_mab_summary.py:75-159](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L75-L159)) — counts how many of the expert `keypoints` are supported by the generated summary. Returns `{"supported_key_points": [...], "recall": N}`.
  3. **Precision** (`_PRECISION_PROMPT_BOOK`, [llm_judge_mab_summary.py:162-196](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L162-L196)) — counts how many sentences in the generated summary are supported by the expert summary. Returns `{"precision": N, "sentence_count": M}`.
- **`_parse_json(text)`** ([llm_judge_mab_summary.py:199-219](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L199-L219)): greedy brace match with `re.findall(r"\{.*?\}", ..., re.DOTALL)`, fallback to ` ```json ``` ` fenced block; returns the *last* JSON object (model often emits an example object in its reasoning before the final score).
- **Aggregation** ([llm_judge_mab_summary.py:266-294](/Users/weichiehhuang/MIRIX_eval/evals/llm_judge_mab_summary.py#L266-L294)): `recall = recall_found / len(keypoints)`, `precision = precision_found / sentence_count`, then
  ```
  F1 = fluency * 2 * recall * precision / (recall + precision)
  ```
  (so a 0 on fluency zeros F1 entirely). Any missing-or-unparseable judge call silently collapses that sub-metric to 0, matching the official rule of skipping `None` outputs. Returns `{fluency, recall_total, recall_found, recall, precision_total, precision_found, precision, f1, raw: {fluency_out, recall_out, precision_out}}`.
- **Book-variant prompts only:** `infbench_sum_eng_shots2` is novel summarization, so `*_book` prompts are used; the official script's lawsuit variants (`multi_lexsum`) aren't in the MAB LRU split.

### Modified files

#### [`evals/organize_results.py`](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py) — 563 lines

The aggregator now supports four judge modes via `--judge {default,mab,substring,mab_summary}` ([organize_results.py:310-326](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L310-L326)). Legacy `--mab-judge` boolean still resolves to `--judge mab` ([organize_results.py:338-339](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L338-L339)).

- **Judge-task tuple is now 9-wide** ([organize_results.py:155-162](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L155-L162)) — `(idx, sample_id, qidx, category, question, expected, predicted, question_id, keypoints)`. `question_id` was added in `ca9bb36` so `judge_task_mab` can detect the `_abs` abstention suffix; `keypoints` was added so `judge_task_mab_summary` can score per-record without re-fetching HF. The other three judges destructure `_keypoints` and ignore it.
- **`judge_task_mab`** ([organize_results.py:242-267](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L242-L267)): if `category in MAB_SUPPORTED_TASKS` route to `evaluate_mab_judge(... abstention='_abs' in question_id ...)`, label `CORRECT`/`WRONG` plus `-abs` suffix when abstention was triggered; unknown categories fall back to the generic LoCoMo judge tagged `WRONG-unknown-cat` so mixed-source runs don't crash.
- **`judge_task_mab_summary`** ([organize_results.py:270-292](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L270-L292)): runs `evaluate_mab_summary_judge(generated_summary=predicted, expert_summary=expected, keypoints=keypoints)`. Per-record `score = F1` (float in `[0,1]`); the four sub-scores live in `details = {fluency, recall_*, precision_*, f1, raw}`. Label is rendered like `F1=0.831 (fluency=1, recall=18/26, precision=22/30)`.
- **Cache invalidation** ([organize_results.py:343](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L343)): `use_cache = (args.judge == "default")` — non-default judges always re-judge (cached labels were produced by a different judge and aren't comparable).
- **Pool sizing** ([organize_results.py:432-437](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L432-L437)): substring runs with `pool_size=1` (local, free), `mab_summary` with `4` (three sequential gpt-4o calls per task — respect rate limits), all other judges `16`.
- **Floats throughout** ([organize_results.py:474-485](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L474-L485)): `total_correct` is a float so `mab_summary`'s per-record F1 averages correctly; for binary judges this still gives an integer count.
- **`summarization_metrics`** ([organize_results.py:515-528](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L515-L528)): only populated when `--judge mab_summary`. Builds `{"gpt-4-fluency": mean, "gpt-4-recall": mean, "gpt-4-precision": mean, "gpt-4-f1": mean}` from per-record `details` and merges into the output `metrics` dict via `**(summarization_metrics or {})` at [organize_results.py:549](/Users/weichiehhuang/MIRIX_eval/evals/organize_results.py#L549) — matching the official `averaged_metrics` shape.

#### [`evals/main_eval.py`](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py)

Switched the LoCoMo memory snapshot from the capped `/memory/components` endpoint to `dump_memories(sample_id)` ([main_eval.py:11](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py#L11) and [main_eval.py:300-311](/Users/weichiehhuang/MIRIX_eval/evals/main_eval.py#L300-L311)). LoCoMo's `_memories.json` token counts now reflect the real PG store, not the 50-per-type / 200-cap truncated view.

#### [`evals/mirix_memory_system.py`](/Users/weichiehhuang/MIRIX_eval/evals/mirix_memory_system.py) — 169 lines

- **`MirixClient(timeout=1800)`** ([mirix_memory_system.py:54](/Users/weichiehhuang/MIRIX_eval/evals/mirix_memory_system.py#L54)) — bumped from the default. With 4096-token chunks (~16k chars), each `/memory/add_sync` takes 3-6 min server-side; 30 min headroom prevents httpx `ReadTimeout` killing the eval mid-ingest.

#### [`evals/task_agent.py`](/Users/weichiehhuang/MIRIX_eval/evals/task_agent.py) — 357 lines

- **`MirixClient(timeout=1800)`** ([task_agent.py:39](/Users/weichiehhuang/MIRIX_eval/evals/task_agent.py#L39)) — same reason; this client is shared with the ingest path via `task_agent.mirix_client`.
- **`_search_memory` hardening** ([task_agent.py:142-190](/Users/weichiehhuang/MIRIX_eval/evals/task_agent.py#L142-L190)):
  - Filter out empty-string / non-string keys before `**params` ([task_agent.py:156](/Users/weichiehhuang/MIRIX_eval/evals/task_agent.py#L156)): `params = {k: v for k, v in params.items() if isinstance(k, str) and k}` — the LLM occasionally produces a tool call with a `""` key, which would `TypeError` on `**params` and kill the whole eval.
  - Wrap `client.search(**params)` in `try/except TypeError` ([task_agent.py:166-173](/Users/weichiehhuang/MIRIX_eval/evals/task_agent.py#L166-L173)) — if the LLM hallucinates a kwarg not in the schema, return `{"success": False, "error": ..., "skipped": True}` instead of crashing. The model sees empty results and tries another query.

### Runner shell scripts

All three follow the same three-stage `[1/3] runner → [2/3] organize_results → [3/3] memory_snapshot` shape, share the same pre-flight (`lsof :8531` + `/health` probe + `datasets` import probe), and end with a per-category summary block and a `psql` count of `episodic / semantic / procedural / resource / knowledge_vault` rows.

#### [`evals/run_mab_longmem_eval.sh`](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_longmem_eval.sh) — 131 lines

- **Pre-flight ([run_mab_longmem_eval.sh:29-48](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_longmem_eval.sh#L29-L48)):** `lsof -ti:8531` must return PID; `urllib.request.urlopen("http://localhost:8531/health", timeout=5).status == 200`; `importlib.util.find_spec("datasets")` must succeed.
- **[1/3] runner ([run_mab_longmem_eval.sh:50-68](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_longmem_eval.sh#L50-L68)):** `TS=$(date +%Y%m%d_%H%M%S); OUT="longmem_mab_${TS}"`. `EXTRA_ARGS` is a bash array: each `MAX_CHUNKS` / `MAX_QS` env var adds `--max-chunks N` / `--max-questions N`. `LIMIT_VAL="${LIMIT:-1}"` default. Invokes `python longmem_eval.py --limit $LIMIT_VAL --run-llm --mirix_config_path ./configs/mab.yaml --output_path $OUT "${EXTRA_ARGS[@]}"`.
- **[2/3] organize_results ([run_mab_longmem_eval.sh:71-74](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_longmem_eval.sh#L71-L74)):** `python organize_results.py --mab-judge "results/longmem/${OUT}"` — full path passed because `organize_results` defaults its fallback to `results/locomo/`.
- **[3/3] snapshot ([run_mab_longmem_eval.sh:77-88](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_longmem_eval.sh#L77-L88)):** `MIRIX_PG_DB="${MIRIX_PG_DB:-mirix}" python memory_snapshot.py save "${OUT}" --agents`, then `cp -R "results/longmem/${OUT}" "snapshots/${OUT}/results"`. Failure is non-fatal — eval + judge already succeeded.
- **Summary:** prints `Acc`, `total_correct/total_judged`, `total_questions`, avg answer latency, avg memory tokens, and per-category accuracy (LongMemEval has `single-session-*`, `multi-session`, `temporal-reasoning`, `knowledge-update`, `single-session-preference`).

#### [`evals/run_mab_ruler_eval.sh`](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_ruler_eval.sh) — 123 lines

- **Pre-flight:** identical to longmem.
- **[1/3] runner ([run_mab_ruler_eval.sh:53-73](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_ruler_eval.sh#L53-L73)):** `OUT="ruler_mab_${TS}"`. Adds `SOURCE_VAL="${SOURCE:-ruler_qa1_197K}"` env var to pick SHDOCQA (default) vs MHDOCQA (`SOURCE=ruler_qa2_421K`). Invokes `python ruler_eval.py --limit $LIMIT_VAL --source "$SOURCE_VAL" --run-llm --mirix_config_path ./configs/mab.yaml --output_path $OUT "${EXTRA_ARGS[@]}"`.
- **[2/3] organize_results ([run_mab_ruler_eval.sh:78](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_ruler_eval.sh#L78)):** hardcoded to `--judge substring` — RULER's official metric is `substring_exact_match`, no LLM judge needed.
- **[3/3] snapshot:** `python memory_snapshot.py save "${OUT}" --agents` then `cp -R "results/ruler/${OUT}" "snapshots/${OUT}/results"`.
- **Summary** omits per-category accuracy — RULER has no `question_types`, so it would be uninformative.

#### [`evals/run_mab_lru_eval.sh`](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_lru_eval.sh) — 147 lines

- **Pre-flight:** identical to longmem.
- **[1/3] runner ([run_mab_lru_eval.sh:62-90](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_lru_eval.sh#L62-L90)):** `OUT="lru_mab_${TS}"`. `SOURCE_VAL="${SOURCE:-infbench_sum_eng_shots2}"` picks `infbench` (default) or `detective_qa`. Same `EXTRA_ARGS` shape.
- **Source-routed judge ([run_mab_lru_eval.sh:71-76](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_lru_eval.sh#L71-L76)):**
  ```bash
  case "$SOURCE_VAL" in
    infbench_sum_eng_shots2) DEFAULT_JUDGE="mab_summary" ;;
    detective_qa)            DEFAULT_JUDGE="substring"   ;;
    *)                       DEFAULT_JUDGE="mab_summary" ;;
  esac
  JUDGE_VAL="${JUDGE:-$DEFAULT_JUDGE}"
  ```
  Overridable via `JUDGE=...`. So infbench → three gpt-4o-2024-05-13 calls per record; detective_qa → free local substring match.
- **[2/3] organize_results ([run_mab_lru_eval.sh:94](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_lru_eval.sh#L94)):** `python organize_results.py --judge "$JUDGE_VAL" "results/lru/${OUT}"`.
- **[3/3] snapshot:** `python memory_snapshot.py save "${OUT}" --agents` then `cp -R "results/lru/${OUT}" "snapshots/${OUT}/results"`.
- **Summary ([run_mab_lru_eval.sh:106-133](/Users/weichiehhuang/MIRIX_eval/evals/run_mab_lru_eval.sh#L106-L133)):** prints `Acc / mean F1`, then iterates `gpt-4-fluency`, `gpt-4-recall`, `gpt-4-precision`, `gpt-4-f1` from the `metrics` block when `mab_summary` was used — surfacing the four official summarization sub-scores.

### Config

#### [`evals/configs/mab.yaml`](/Users/weichiehhuang/MIRIX_eval/evals/configs/mab.yaml) — 46 lines

Standalone MAB profile, independent from the LoCoMo `0201c` profile so tuning can diverge.

- **`llm_config`** ([mab.yaml:9-14](/Users/weichiehhuang/MIRIX_eval/evals/configs/mab.yaml#L9-L14)): `model: "gpt-4.1-mini"`, `model_endpoint_type: openai`, `model_endpoint: https://api.openai.com/v1`, `context_window: 128000`. This is the answer LLM and the per-agent memory-update LLM. `api_key: your-api-key` is the placeholder substituted at load time by `_resolve_api_keys` in [mirix_memory_system.py](/Users/weichiehhuang/MIRIX_eval/evals/mirix_memory_system.py#L31-L44) using `OPENAI_API_KEY` from `.env`.
- **`topic_extraction_llm_config`** ([mab.yaml:16-21](/Users/weichiehhuang/MIRIX_eval/evals/configs/mab.yaml#L16-L21)): `model: "gpt-4.1-nano"`, same `openai` shape — used for the cheap topic-extraction pass before routing into per-agent memory operations.
- **`embedding_config`** ([mab.yaml:23-28](/Users/weichiehhuang/MIRIX_eval/evals/configs/mab.yaml#L23-L28)): `text-embedding-3-small`, dim `1536`. `build_embeddings_for_memory: true` ([mab.yaml:30](/Users/weichiehhuang/MIRIX_eval/evals/configs/mab.yaml#L30)) — embeddings are built at ingest so `search_method=embedding` in `task_agent._search_memory` ([task_agent.py:159-160](/Users/weichiehhuang/MIRIX_eval/evals/task_agent.py#L159-L160)) has vectors to hit.
- **`meta_agent_config`** ([mab.yaml:32-46](/Users/weichiehhuang/MIRIX_eval/evals/configs/mab.yaml#L32-L46)): `system_prompts_folder: prompts/0201a/`. Agents: `core_memory_agent, resource_memory_agent, semantic_memory_agent, episodic_memory_agent, procedural_memory_agent, knowledge_vault_memory_agent`. Initial core memory ships with `label: "human"` (empty value) and `label: "persona"` (`"I am a helpful assistant."`).
- **`chaining=false` (not in YAML — set per-call):** `MirixMemorySystem.add_chunk` ([mirix_memory_system.py:82](/Users/weichiehhuang/MIRIX_eval/evals/mirix_memory_system.py#L82)) hardcodes `chaining=False` on every `/memory/add_sync` call so the multi-agent chain doesn't run on every session, plus `filter_tags={"scope": "read_write", "kind": "conversation_session"}`. That's the eval-only knob that keeps ingest from blowing past the timeout budget.

## MIRIX core patches

Exhaustive reference for commit `b27fe93` ("MIRIX: Redis modules-missing latch + small robustness fixes") at `/Users/weichiehhuang/MIRIX_eval`. Three files changed, all additive — `+68/-2` lines total.

---

### `mirix/database/redis_client.py` — Redis modules-missing latch

**What's added:**
- New instance attribute `self._modules_missing = False` initialized in `RedisMemoryClient.__init__` (just after `self.client = Redis(connection_pool=self.pool)`).
- New private helper `_check_modules_missing(exc: Exception) -> bool` that returns `True` iff `"unknown command"` appears in `str(exc).lower()`, and on the first hit emits one `logger.warning(...)` and sets `self._modules_missing = True`.
- 7 callers get an entry guard `if self._modules_missing: return <sentinel>` and their `except Exception as e:` blocks call the helper before falling through to the original `logger.warning/error` line:
  - `set_json` (sentinel: `return False`)
  - `search_text` (sentinel: `return []`)
  - `search_vector` (sentinel: `return []`)
  - `search_recent` (sentinel: `return []`)
  - `search_recent_by_org` (sentinel: `return []`)
  - `search_vector_by_org` (sentinel: `return []`)
  - `search_text_by_org` (sentinel: `return []`)

**Why:** stock OSS Redis without RedisJSON + RediSearch modules raised `ResponseError("unknown command 'JSON.SET'" / "'FT.SEARCH'")` on every cache write/query. A full SHDOCQA ingest emitted thousands of ERROR lines. First failure now logs a single WARNING and flips the latch; every subsequent call returns its empty sentinel silently.

**Behavior delta:** the PG fallback path is identical. Managers' `if results: ...` branch still sees `[]` and falls through to PostgreSQL exactly as before — only log volume changes.

**Risk:** the latch can only flip when the exception's `str(...)` contains `"unknown command"`. Real Redis `ResponseError`s for other reasons (auth, connection, OOM, etc.) continue down the original error-log path unchanged.

**Verification:** full SHDOCQA Q&A re-run on 2026-06-03 emitted exactly 1 WARNING line (`"Redis JSON/Search modules not loaded — skipping Redis cache writes/queries for the rest of this process"`) then silence for the rest of the run.

**Before vs After (representative — `set_json`; the same shape applies to all 7 callers):**

Before:
```python
async def set_json(self, key: str, data: dict, ttl: Optional[int] = None) -> bool:
    try:
        await self.client.json().set(key, "$", data)
        ...
        return True
    except Exception as e:
        logger.error("Failed to set JSON for %s: %s", key, e)
        return False
```

After:
```python
async def set_json(self, key: str, data: dict, ttl: Optional[int] = None) -> bool:
    if self._modules_missing:
        return False
    try:
        await self.client.json().set(key, "$", data)
        ...
        return True
    except Exception as e:
        if self._check_modules_missing(e):
            return False
        logger.error("Failed to set JSON for %s: %s", key, e)
        return False
```

New helper (added once, used by all 7 callers):
```python
def _check_modules_missing(self, exc: Exception) -> bool:
    if "unknown command" in str(exc).lower():
        if not self._modules_missing:
            logger.warning(
                "Redis JSON/Search modules not loaded — skipping "
                "Redis cache writes/queries for the rest of this "
                "process. PG remains authoritative. Original: %s",
                exc,
            )
            self._modules_missing = True
        return True
    return False
```

**No class, method signature, or public API changed** — the patch only adds one private attribute, one private helper method, and intra-method guards. All public methods keep their existing names, parameter lists, and return types/sentinels.

---

### `mirix/functions/function_sets/memory_tools.py` — defensive dict access in `semantic_memory_insert`

**What's added:** a single character-level change at line 657 inside the per-item construction loop of `semantic_memory_insert`.

**Exact change:** `source=item["source"]` → `source=item.get("source", "")`.

**Why:** an LLM tool call for `semantic_memory_insert` occasionally omits the optional `source` field; the bracketed access raised `KeyError` and aborted the rest of the items in that batch. Observed on RULER ingest where ~5% of semantic items had a missing `source` during the original SHDOCQA full run.

**Risk:** zero — `source` was always optional at the storage layer; an empty string is preferable to a hard crash mid-batch.

**Before vs After:**

Before:
```python
name=item["name"],
summary=item["summary"],
details=item["details"],
source=item["source"],
organization_id=self.actor.organization_id,
```

After:
```python
name=item["name"],
summary=item["summary"],
details=item["details"],
source=item.get("source", ""),
organization_id=self.actor.organization_id,
```

**No class, method signature, or public API changed** — only the dict-access expression for one optional field inside the existing function body was softened.

---

### `mirix/client/remote_client.py` — per-request timeout plumbing + corrected comment

**What's added:** two changes inside `MirixClient._request()`, immediately around the `await self._client.request(...)` call.

1. **Timeout kwarg:** added `timeout=self.timeout` to the `request(...)` call so the per-request budget is plumbed through `request.extensions['timeout']`.
2. **Comment rewrite:** the old comment claimed `AsyncClient(timeout=N)` was silently overridden by the wrapped `AsyncHTTPTransport`. A fake-transport repro (see the changelog's lessons-learned section) showed `RetryTransport` propagates `request.extensions['timeout']` unchanged. The new comment frames `self.timeout` as the caller-controlled budget the per-request argument reads from.

**Why the timeout kwarg:** without it the request inherited only the `AsyncClient.timeout` configured at construction. The eval was constructing `MirixClient()` without an explicit timeout, so `self.timeout` stayed at the 60s default and big chunks (>60s server-side) timed out. Passing `timeout=self.timeout` lets callers raise it via `MirixClient(timeout=...)` without rebuilding the underlying `AsyncClient`.

**Why the comment rewrite:** technical accuracy — `RetryTransport` does not strip the per-request timeout; it forwards `request.extensions['timeout']` to the wrapped `AsyncHTTPTransport` unchanged, so the previous "silently overridden" claim was wrong.

**Risk:** zero — only adds a kwarg sourced from the existing client's own `self.timeout` attribute. Pre-existing callers that constructed `MirixClient()` without an explicit timeout still get the 60s default.

**Before vs After:**

Before:
```python
response = await self._client.request(
    method=method, url=url, json=json, params=params, headers=headers
)
```

After:
```python
# Per-request timeout is set from self.timeout so a caller can
# raise it via MirixClient(timeout=...) without rebuilding the
# AsyncClient. RetryTransport propagates request.extensions
# ['timeout'] to the wrapped AsyncHTTPTransport unchanged.
response = await self._client.request(
    method=method, url=url, json=json, params=params, headers=headers,
    timeout=self.timeout,
)
```

**No class, method signature, or public API changed** — `MirixClient._request` keeps its existing parameters and return type; only one keyword argument is forwarded to the underlying `httpx` call, and an adjacent comment was corrected.

---

**Summary across all 3 patches:** every change is strictly additive. No class hierarchy, method signature, parameter name, return type, sentinel value, or public/imported symbol was renamed, removed, or repurposed. The patches reduce log noise (Redis), prevent a single-item crash from aborting an ingest batch (semantic memory), and let the caller's timeout budget reach the wire (remote client) — without touching anything callers depend on.

## Benchmark results

### SHDOCQA full runs (1 conversation × 100 questions, `substring_exact_match` judge)

| date | chunking | n_chunks | acc | total_ingest_min | per-chunk_median_s | notes |
|---|---|---|---|---|---|---|
| 2026-06-01 | char-4096 | 273 | 82% | 169 | 32.6 | baseline |
| 2026-06-02 | token-4096 | 50 | 27% | 97 | 111.8 | **invalid — broken retrieval (`user.organization_id` NULL bug). Do not cite.** |
| 2026-06-03 | token-4096 | 50 | 69% | 97 | 111.8 | true apples-to-apples vs row 1 |

Net: char-4096 wins on accuracy by 13 pp; token-4096 wins on ingest wall-clock by 1.74×.

### Smoke tests (1 chunk × 1 question per dataset, pipeline validation)

| dataset | retrieval block size | predicted answer (truncated) | result |
|---|---|---|---|
| SHDOCQA | 2424 chars | "Normandy is located in France." | CORRECT |
| MHDOCQA | 1777 chars | "Scott Derrickson and Ed Wood were both American." | substring miss (semantically right but answer schema wants literal "yes") |
| LongMemEval-S | 1829 chars | "There is no clear information about how many hours you work..." | appropriate abstention (single-session smoke can't contain work-hours) |
| LRU/infbench | 2524 chars | "Miss Jennifer Pete is a young woman of plain but impressive beauty..." | F1=0, fluency=1 — coherent but 1-of-95-chunks ingestion limits coverage |
| LRU/detective | 258 chars | "{answer: D. Miss House}" | substring miss (wrong option) but confident, retrieval working |

### Memory store characterisation (SHDOCQA, 50 token-4096 chunks → 1040 epi + 1194 sem items)

- Proper-noun density per semantic item: 18.77 (char) → 15.39 (token); -18%
- Year-token density per semantic item: 1.51 → 1.22; -19%
- Mean summary length: ~163 chars (char) vs ~152 chars (token); -7%
- Mean details length: ~457 chars (char) vs ~393 chars (token); -14%
- Distinct named entities: 6368 (char) vs 5715 (token); 1104 entities only in char store, 451 only in token store, 5264 in both

### Cost / efficiency (SHDOCQA, same row)

- Total ingest time: 10131s (char) vs 5832s (token); 1.74× speedup
- Per-chunk median time: 32.6s (char) vs 111.8s (token); 3.4× heavier per call
- Per-correct-answer cost: 124s (char) vs 85s (token); -32% (token gets more accuracy per unit time but lower absolute accuracy)
- Average answer latency at retrieval time: 2.82s (char) vs 2.54s (token); similar

### System health (all 4 dataset smokes on 2026-06-03 / -04)

- Server uptime: 1d 14h continuous
- 5 user records (`shdocqa_ruler_qa_0`, `mhdocqa_ruler_qa_0`, `longmem_s_0`, `infbench_0`, `conv-26`) all with `organization_id='mirix-eval-org'`
- Redis user cache consistent with PG
- `retrieve_with_conversation` returns non-empty `episodic_memory` block for every smoke
- Zero `httpx.ReadTimeout`, zero Q&A crashes after the empty-kwarg patch

## Bugs discovered + lessons learned

A chronicle of the rabbit holes hit while standing up the MAB eval suite. Read this before debugging anything that smells like one of these — most of them mimic "model got worse" or "framework bug" but are something else entirely.

### 1. macOS `tmp_cleaner` wiping `/tmp/mirix_isolate/.venv` at midnight

- **Symptom:** A SHDOCQA full run failed mid-stream with `FileNotFoundError [Errno 2]` thrown from inside `httpx`/`ssl` while calling OpenAI embeddings. The client log showed `/memory/add_sync` returning HTTP 500 from the server. Up until ~00:00 the run was healthy; after midnight every request failed identically.
- **Root cause:** `/usr/libexec/tmp_cleaner` (LaunchDaemon `com.apple.tmp_cleaner`, `StartCalendarInterval` Hour=0) wiped 76 packages under `/tmp/mirix_isolate/.venv/lib/python3.12/site-packages/` at 00:00 on 2026-06-02. `certifi/cacert.pem` was among the casualties, so `ssl.create_default_context(cafile=certifi.where())` raised `FileNotFoundError` on the very next outbound HTTPS call.
- **Diagnostic path:**
  1. Server log pinpointed the throw to `.../mirix/embeddings.py:421`.
  2. `ls -la` inside the venv showed 76 empty package directories, all with `mtime` of `2026-06-02 00:00:0X`.
  3. `cat /System/Library/LaunchDaemons/com.apple.tmp_cleaner.plist` confirmed the midnight schedule and the `/tmp` scope.
- **Fix:** Moved the whole `MIRIX_eval` working tree out of `/tmp` to `~/MIRIX_eval`, recreated the venv there, restarted the server. No more midnight reaper.
- **Forward-looking advice:** Never put a long-lived eval venv under `/tmp` on macOS. If something forces a tmp-ish path, use `/private/var/tmp` or another location outside `tmp_cleaner`'s scope. Add a guard at the top of every runner script:
  ```bash
  [[ "$(realpath .venv)" != /tmp/* ]] || { echo "venv is under /tmp; tmp_cleaner will eat it"; exit 1; }
  ```

### 2. `users.organization_id` NULL silently kills retrieval (and the Redis user cache hides the fix)

- **Symptom:** A new SHDOCQA full run with token-4096 chunking scored 27%, down 55 pp from char-4096's 82%. The model produced abstentions ("no specific information available") for most questions, as if its memory had been wiped.
- **Confounder:** Two unrelated problems were stacked. The bigger one was that retrieval was returning zero items for every question — completely independent of the chunker change.
- **Root cause:** I'd migrated the user row `ruler_qa_0` → `shdocqa_ruler_qa_0` with a raw `INSERT INTO users SELECT ... FROM users WHERE id=...` that did not list `organization_id`. The new row's `organization_id` was NULL. Every `list_*_items` query filters `WHERE organization_id = user.organization_id`, which becomes `WHERE organization_id IS NULL` — no row in the memory tables matches, so retrieval returns empty.
- **Second-level confounder:** `UserManager.get_user_by_id` caches the Pydantic user in Redis via plain `HGETALL` (not RedisJSON, so unaffected by the modules-missing latch). A bare PG `UPDATE` does not invalidate that cache. After `UPDATE users SET organization_id='mirix-eval-org'`, `get_user_by_id` was still returning `organization_id=None` from cache, so retrieval kept returning empty.
- **Diagnostic path:**
  1. `retrieve_with_conversation` reported `total_count=1194` but `items=0` for semantic memory — the smoking gun (data exists, filter excludes it).
  2. Manager-level `list_semantic_items(use_cache=False)` also returned 0, ruling out item-level cache.
  3. Raw ORM `SELECT` against `users` showed `organization_id='mirix-eval-org'`.
  4. `UserManager.get_user_by_id` Pydantic still returned `organization_id=None`.
  5. `redis-cli HGETALL user:shdocqa_ruler_qa_0` showed the `organization_id` field literally empty.
- **Fix:**
  ```sql
  UPDATE users SET organization_id='mirix-eval-org' WHERE id='shdocqa_ruler_qa_0';
  ```
  ```bash
  redis-cli DEL user:shdocqa_ruler_qa_0
  ```
- **Forward-looking advice:**
  - Any raw-SQL user migration MUST copy `organization_id`. The Pydantic model defaults it to `DEFAULT_ORG_ID`, but raw `INSERT ... SELECT` will happily leave it NULL.
  - Smoke test after every user mutation: `curl '/memory/components?user_id=X&memory_type=semantic&limit=3'`. If `total_count > 0` but `items == []`, you are in this trap.
  - Anywhere DB user rows are mutated programmatically, either invalidate `user:<id>` in Redis or write through. A `DEL user:<id>` after any user UPDATE is cheap insurance.

### 3. Char-4096 vs token-4096 chunking — apples-to-apples is harder than it looks

- **Symptom:** SHDOCQA scores read: char-4096 = 82%, token-4096 (broken) = 27%, token-4096 (fixed) = 69%. Taken at face value, the first comparison "proved" the chunker change was catastrophic. It wasn't — the 27% was Bug #2.
- **Root cause (the real, post-fix finding):** Token-4096 chunks pack roughly 3.4× more text per call than char-4096, but the per-chunk extraction budget (~45 items) stayed constant. The extractor compensates by summarising at a higher abstraction level. Specific entities (Conrad of Montferrat, Maciot de Bethencourt, etc.) get folded into thematic items or dropped entirely.
- **Numbers:** Proper-noun density dropped from 18.77 → 15.39 per item (-18%). 1,104 distinct entities present in the char-4096 store had no counterpart in the token-4096 store.
- **Diagnostic path:** Once Bug #2 was fixed and the new score landed at 69%, comparing entity-density and entity-set diffs between the two stores explained the remaining ~13 pp gap.
- **Forward-looking advice:**
  - When changing two things at once (chunker + user_id), prove each one independently. Re-run the question phase with the new chunker *and* the old chunker against the same `user_id` before drawing conclusions about chunking.
  - For needle-retrieval evals (RULER, LongMemEval-S short-answer), prefer the smaller chunk. For summarisation workloads (LRU, InfBench), the tradeoff may invert — bigger chunks may help by giving the extractor more context to abstract over.

### 4. `RetryTransport` timeout propagation — the prior comment was wrong

- **Symptom:** Eval calls were silently dying at the 60s mark even though `MirixClient(timeout=1800)` was nominally being used. A comment in `remote_client.py` blamed the framework:
  > "The wrapped `RetryTransport` otherwise falls back to `httpx.AsyncHTTPTransport`'s default 60s for read/write, which silently overrides `AsyncClient(timeout=N)`."
- **Root cause:** That comment is wrong. A minimal fake-transport repro showed that `RetryTransport.handle_async_request` forwards the request unchanged to `self._wrapped.handle_async_request`, and the wrapped transport reads `request.extensions['timeout']` exactly as it would without the wrapper. `AsyncClient(timeout=N)` and per-request `client.request(..., timeout=N)` both propagate correctly through the wrapper. The real cause of the eval's 60s ceiling: the eval was constructing `MirixClient()` *without* passing `timeout`, so `self.timeout` stayed at the default 60s, and the per-request `timeout=self.timeout` then enforced 60s.
- **Diagnostic path:** Built a fake `httpx.AsyncBaseTransport` that recorded the resolved timeout from each incoming request and verified it matched the configured value in every case — including through `RetryTransport`.
- **Fix:** Pass `timeout=1800` when constructing `MirixClient` in both `mirix_memory_system` and `task_agent`.
- **Forward-looking advice:** When a "framework bug" claim lives in a code comment, re-verify with a minimal repro before trusting it. Comments age out faster than code, and the "obvious" framework bug is almost always a misconfiguration one level up.

### 5. Eval crashed at Q57 from an empty-string kwarg key in an LLM tool call

- **Symptom:** SHDOCQA Q&A run died with `TypeError: MirixClient.search() got an unexpected keyword argument ''` after answering 56 of 100 questions. The first 56 questions were fine; Q57 happened to elicit a tool call whose JSON args contained an empty-string key.
- **Root cause:** `task_agent.py:_search_memory` does `mirix_client.search(user_id=..., **params)` where `params` is the JSON-decoded args dict from an OpenAI tool call. The LLM occasionally emits a key that is the empty string (`""`). Python rejects an empty kwarg name when it reaches the receiver's signature, and the whole eval falls over.
- **Fix:** Filter falsy/non-string keys before the splat, plus wrap the call in a `try/except TypeError` so the model can retry with a cleaner query:
  ```python
  params = {k: v for k, v in params.items() if isinstance(k, str) and k}
  try:
      result = mirix_client.search(user_id=user_id, **params)
  except TypeError:
      return {"success": False, "skipped": True, "reason": "bad tool args"}
  ```
- **Forward-looking advice:** Any `**dict` splat of LLM-provided args needs (a) a key-shape filter and (b) a `TypeError` safety net. Cost: one `if`-statement. Benefit: the eval doesn't die ~halfway through a 100-question run because the model emitted one weird tool call.

---

**Meta-lesson across all five:** Most of these masqueraded as "model got worse" or "framework is broken." In every case the real cause was lower in the stack — a LaunchDaemon, a missing column, a stale Redis row, a misread comment, an unsanitised splat. Before blaming the model or the framework, prove the eval's own substrate (filesystem, DB rows, cache, client config, tool-args plumbing) is intact.

## Next steps

- Push branch to origin: `git push --force-with-lease origin mab_runner`. The backup branch `mab_runner_pre_squash_260604-1402` is a local-only safety net preserving the pre-squash 14-commit history; it stays unpushed.
- Optional full runs queued behind validated smokes:
  - MHDOCQA — 100 questions on the multi-hop RULER QA2 subset
  - LongMemEval-S — 500 questions across the full single-session-* / multi-session / temporal-reasoning / knowledge-update / single-session-preference categories
  - LRU / infbench — 100 questions × ~95 chunks each (summarisation workload, three gpt-4o-2024-05-13 calls per record at judge time)
- Future investigation: the chunk-size tradeoff measured on SHDOCQA (char-4096 +13 pp vs token-4096) is a needle-retrieval finding. Summarisation workloads (LRU/infbench) may invert it because the extractor benefits from more context per call. Re-measure with an infbench full run to see whether token-4096 wins on F1 there.
- Future infra: add write-through invalidation of the Redis user cache (`user:<id>`) whenever a DB user row is mutated by any direct SQL path — closes the foot-gun that hid the `organization_id` NULL fix during Bug #2.
