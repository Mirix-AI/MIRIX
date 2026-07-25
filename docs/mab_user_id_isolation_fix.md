# Patch note: per-sub_dataset user_id isolation + memory purge for the MAB adapter

**Scope.** `samples/memoryagentbench/mirix_adapter.py`,
`samples/memoryagentbench/run_bench.py`,
`samples/memoryagentbench/configs/mirix_gpt-4o-mini.yaml`. No changes to
MIRIX core or MAB.

**Status.** Bug fix. Every MAB result produced before this patch is
contaminated (see "Impact" below) and must be re-run.

## The bug

The MAB adapter wrote every benchmark's memory into the **same MIRIX
`user_id`**. Three things combined to make this silently corrupt results:

1. **A single shared user_id.** The adapter computed `user_id` from
   `mirix_user_prefix`, which the config set to a constant (`mab`). So
   every sub_dataset and every context resolved to `mab-ctx0`.

2. **`add` is purely additive.** MIRIX's `/memory/add` never replaces;
   it appends. Nothing ever cleared prior memory.

3. **`--force` did not force a re-ingest.** `--force` deleted the result
   JSON and skipped the "context already complete" check, but the
   per-context agent-state sentinel folder was left on disk. The runner
   then hit `if os.path.exists(save_folder): agent.load_agent()` and
   reused the stale server-side memory instead of re-ingesting.

The net effect: every MAB run accumulated on top of every previous run's
memory, under one user_id, with no way to reset.

### How it surfaced

A LongMemEval-S* run scored 10%. Inspecting a failing question
("How long have I had my cat, Luna?") showed `retrieve_with_conversation`
returning, as its top episodic item:

> "User shared further extensive learned factual data, expanding the list
>  to over 18331 items covering ... American Locomotive Company was
>  created in the country of Soviet Union; Taoism was founded by Juliette
>  Gordon Low; ..."

That is FactConsolidation's `sh_262k` data. It had been ingested into
`mab-ctx0` by an earlier run, was newer than the LongMemEval episodics,
and so dominated the `recent` ordering and flooded the retrieval window.
The actual cat-Luna memory (`ep_AVF5`) never made it into the top-10.
The LongMemEval run was effectively being evaluated against a memory
store that was ~99% unrelated FactConsolidation facts.

## Impact

Contaminated — must be re-run:

- FactConsolidation `sh_6k` (off / auto), `sh_32k`, `sh_64k`, `sh_262k`
- FactConsolidation `mh_6k`
- LongMemEval-S* (1 sample)

`sh_6k` was the first MAB run and may have started against an empty
store, but the sweep that followed (`sh_32k` → `sh_64k` → `sh_262k`)
each ran on top of all prior sub_datasets' memory, so even the
FactConsolidation length-sweep numbers are not trustworthy.

Not affected: all LoCoMo results. The LoCoMo pipeline
(`evals/main_eval.py` via `eval_locomo_single.py`) already uses a
per-sample `user_id` (`locomo-user-<sample_id>`), so its 10-conversation
run was never cross-contaminated.

## The fix

### 1. user_id is namespaced by sub_dataset and not configurable

`mirix_adapter.py` — `_user_prefix` is now hard-coded:

```python
# user_id is ALWAYS namespaced by sub_dataset. This is deliberately not
# configurable ...
self._user_prefix = f"mab-{self.sub_dataset}"
```

`_user_id_for_context` then yields `mab-<sub_dataset>-ctx<N>`. Each
sub_dataset gets its own user_id space; each context gets its own
user_id within it. The `mirix_user_prefix` config key is removed.

### 2. Server-side memory is purged before the first ingest

`mirix_adapter.py` — new `_purge_user_memory(user_id)`:

```python
def _purge_user_memory(self, user_id: str) -> None:
    if user_id in self._purged_user_ids:
        return
    self._purged_user_ids.add(user_id)
    try:
        self._run(self._client._request("DELETE", f"/users/{user_id}/memories"))
    except Exception as exc:
        # 404 just means the user has no memory yet — fine.
        ...
```

It calls the existing server endpoint `DELETE /users/{user_id}/memories`
(hard-deletes all episodic / semantic / procedural / resource /
knowledge-vault memory, messages and blocks for the user; preserves the
user record). `_purged_user_ids` guards it so it fires at most once per
user_id per process. `_memorize` calls it before its first `add` for a
user, so every ingest starts from a clean slate.

### 3. `--force` now actually forces a re-ingest

`run_bench.py` — before constructing the adapter for a context:

```python
if args.force and os.path.isdir(save_folder):
    shutil.rmtree(save_folder, ignore_errors=True)
```

Dropping the local sentinel folder makes the runner take the
`_memorize` path instead of `load_agent`, which in turn triggers the
server-side purge from fix #2. Without this, `--force` would re-create
the result file but keep reusing stale server memory.

`run_ablation.py` needs no change: it spawns `run_bench.py` with
`--force`, so it inherits the corrected behaviour.

### 4. Config cleanup

`configs/mirix_gpt-4o-mini.yaml` — the `mirix_user_prefix: mab` line is
removed and replaced with a comment explaining that user_id is not
configurable and that memory is purged before re-ingest.

## Behaviour after the patch

- `sh_6k` writes to `mab-factconsolidation_sh_6k-ctx0`, `sh_32k` to
  `mab-factconsolidation_sh_32k-ctx0`, LongMemEval-S* to
  `mab-longmemeval_s*-ctx0`, and so on — no cross-talk.
- The first ingest into any user_id hard-deletes whatever memory was
  there, so re-runs start clean.
- `--force` is a true from-scratch re-ingest.

## Follow-ups (not done in this patch)

- The legacy `mab-ctx0` user still holds the ~25k contaminated mixed
  memories from pre-patch runs. It can be purged with
  `DELETE /users/mab-ctx0/memories`; new runs no longer touch it.
- All MAB benchmarks (FactConsolidation sweep, `mh_6k`, LongMemEval-S*)
  need to be re-run; the pre-patch numbers in
  `evals/results/mab/.../RESULTS.md` should be regenerated.
