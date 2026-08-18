# evals/harness — the scripts that produced the numbers in HANDOVER.md

These lived only in `/home/lj/MIRIX_eval/` and were referenced by absolute path from the
handover, so a clone of this repository had a runbook pointing at files it did not contain.
They are here now. Paths inside them still point at `/home/lj/MIRIX_eval` and
`/home/lj/code/MIRIX` — fix those before running elsewhere.

Roughly eighty more scripts remain in that directory. Most are one-off probes from a single
afternoon and are not worth carrying; what is here is the subset that produced a result
someone else is expected to build on.

## Run scripts

| file | what it does |
|---|---|
| `locomo_clean.sh` | full LoCoMo ingest + QA, clean prompts. Copy this for a new ingest arm. |
| `graph_ablation.sh` | four QA-only arms over ONE store. **Copy this for any retrieval or answering arm** — it is the template that removes ingest drift. |
| `p3_nograph.sh` | the no-graph arm on three conversations |
| `v71_shdoc_chunk_ab.sh` | SHDocQA on the v7.1 branch, token vs char chunking |

## Measurement

| file | what it measures |
|---|---|
| `hypermem_judge.py` | HyperMem's grader, reproduced exactly including its 2-of-3 majority. Re-scores any stored run without re-answering. |
| `compare_runs.py` | paired comparison of two runs |
| `window_recall.py` | how much of the source's distinctive vocabulary survives into the store. **Judge-free and deterministic** — the model for any metric that has to be readable at n=1. |

## Error analysis

| file | finding it produced |
|---|---|
| `final_triage.py` + `final_triage.json` | the four-bucket split of all 169 errors, with a positive control at every stage. Ingest 60, gold 41, retrieval 38, answering 30. |
| `verify_had_everything.py` | killed the earlier claim that 83 errors were the answerer's fault — a reader model puts only 12% of them there |
| `why_evidence_missing.py` | separates "never written" from "written in different words" |
| `error_triage.py` | the earlier token-overlap triage, kept because its failure mode is instructive |
| `oracle_ceiling.py` | the retrieval ceiling: an oracle retriever rescues 67 of 169 errors, so 102 are beyond any ranking work |
| `merge_would_connect.py` / `anchor_merge_sim.py` | merging 1829 near-duplicate anchors connects the evidence for 1 of 31 failing multi-hop questions |
| `coldfact_gate.py` | the offline gate that killed the cold-fact lane in twenty minutes instead of three hours of QA |

## Answering-side replays

`replay_ab.py`, `replay_disc.py`, `replay_format.py` re-send stored contexts with the
answering step changed, so retrieval variance is excluded. Between them they hold the eleven
interventions that all failed: reworded format rules, naming the failure mode, forced
citation, select-then-answer, self-consistency, verify-then-retry, WHO/WHEN/WHAT tables.
Every one gains 4-11 on the 110 target questions and loses more on the 1335 already right.

Read those before proposing a twelfth.

## The one habit worth copying

Every arm in these scripts ends by printing a verification line — rows under the expected
prefix, rows outside it, chunks ingested, whether the code path under test actually fired.
Three runs in this project returned a plausible number while not running the experiment at
all, and one burned six hours before the verification line caught it. Assert early, not at
the end.
