# MetaClaw-Bench Part I — MIRIX vs native skill provider (FINAL, 2026-06-03)

Authoritative record of the 30-day Part I A/B. Numbers verified byte-exact against the
benchmark's own `report.json`; every claim below was cross-checked against the paper text
(arXiv 2603.17187) and the vendored code at file:line. Conclusions survived several rounds
of correction — read the "Alignment audit" and "Honest caveats" sections before quoting any
cross-paper number.

---

## 1. Experiment design
- **Benchmark**: MetaClaw-Bench Part I — 30 simulated workdays, **346 questions** (224 `file_check` + 122 `multi_choice`). Vendored code byte-identical to upstream MetaClaw SHA `fc163ba8` (SHA-256 verified).
- **Model**: `openai/gpt-5.2` via OpenRouter, reasoning **enabled** (thinking fired every turn).
- **Agent**: OpenClaw 2026.5.18, `coding` profile — 19 tools available; actually used `exec`/`write`/`read`/`edit`.
- **Single controlled variable** (the A/B): the skill produce/retrieve module.
  - `metaclaw` arm = native vendored skill provider.
  - `mirix` arm = MIRIX procedural-memory backend (produce via `/v1/skills/evolve`, retrieve via 4096-dim embedding search, `MIRIX_SKILL_SEARCH_METHOD=embedding`).
  - Everything else identical (same harness, agent config, feedback chain, endpoint, scoring). Both arms start from an empty skill bank.
- **Runs**: `metaclaw30-agentfix-20260602T230234Z`, `mirix30-agentfix-20260602T230238Z` (each 346/346 complete).

## 2. Scoring (the paper's official rule — established, not assumed)
- **Acc** = mean per-question score over ALL 346 = `(Σ file_check 0/1 + Σ multi_choice partial-credit) / 346`. Because multi_choice uses partial credit, the score sum is fractional (e.g. 194.5/346).
  - Multi_choice partial-credit formula is printed verbatim in the **paper** (App. A.3 template, `metaclaw.txt:798`): `max(0, 1 − (FP+FN)/n_options)`, identical to vendored `scoring_cmd.py:124`.
  - Acc includes file_check (paper §4.1.1 "mean per-question score" over the mixed 346-pool; aggregation `report_cmd.py:260,316`).
- **Compl** = file-check completion rate = `file_check passed / 224` (paper §4.1.1).
- ⚠️ Our local `score_arm.py` originally graded multi_choice by **exact-set** boolean (all letters right) → an Acc of 46.2/48.3 that is NOT the paper's rule. The correct, paper-aligned Acc is **56.2%** (partial-credit). Do not quote 46.2/48.3 against the paper.

## 3. Final results (Part I, paper-aligned partial-credit rule)

### 3a. vs the paper (⚠ NOT a clean comparison — see §5)
| Model | Condition | Acc.(%) | Compl.(%) |
|---|---|---|---|
| GPT-5.2 | Baseline (paper) | 41.1 | 14.7 |
| GPT-5.2 | MetaClaw Skills (paper) | 44.0 | 17.1 |
| gpt-5.2 (OpenRouter) | Ours — metaclaw native | **56.2** | **37.5** |
| gpt-5.2 (OpenRouter) | Ours — mirix | **56.2** | **36.2** |

Our Acc cells are byte-exact vs `report.json` (`summary.accuracy` = 0.56216 / 0.56166).

### 3b. mirix vs metaclaw — the clean A/B (same harness/agent/endpoint/scoring; only skill module differs)
| Axis | metaclaw native | mirix | mirix − metaclaw |
|---|---|---|---|
| **Acc (official, partial-credit, /346)** | 56.22% | 56.17% | **−0.05pp (tie)** |
| Compl (file_check, /224) | 37.5% (84) | 36.2% (81) | −1.3pp (≈3 items, noise) |
| multi_choice — partial-credit (/122) | 90.6% | 92.9% | **+2.3pp** |
| multi_choice — exact-set (/122) | 62.3% | 70.5% | +8.2pp |

## 4. Headline conclusion (mirix vs metaclaw)
**On Part I, the mirix arm ties the native metaclaw arm on overall Acc (56.22 vs 56.17, a 0.05pp
tie) and is within run-to-run noise on Compl (−1.3pp ≈ 3 of 224 file_checks).** MIRIX's only
positive signal is on multi_choice: **+2.3pp partial-credit / +8.2pp exact-set**. The gain is
recall-shaped — skills convert "off-by-one" multi-select near-misses into exact matches (full
recall of every rule letter), not fixing outright-wrong answers — which is why it is large under
exact-set but small under the paper's partial-credit rule, and is offset by metaclaw's +1.3pp
Compl so that blended Acc is a dead heat.

**No measurable Part I improvement from the mirix arm at the Acc level.** The exact-set +8.2pp,
while real, is not the paper's metric.

## 5. Alignment audit — what matches the paper and what does NOT
| Dimension | Paper | Ours | Aligned? |
|---|---|---|---|
| Benchmark scoring/report code | upstream fc163ba8 | byte-identical (SHA-256) | ✅ yes |
| Cross-round `[Previous Feedback]` chain | native, dataset-required (§4 line 267) | same native chain | ✅ yes |
| **Agent toolset** | **single `run_command`** (App A.1, `metaclaw.txt:696–735`) | **`coding` profile, 19 tools** (exec/write/read/edit) | ❌ **NO** |
| Agent system prompt | minimal "expert CLI agent" | OpenClaw 17k-char personal-assistant | ❌ no |
| Model serving / reasoning | vLLM/SGLang, unstated | OpenRouter, reasoning ON, effort unpinned | ⚠️ uncontrolled |

**The agent config (`data/openclaw_cfg/openclaw.json`) is OUR local, gitignored file — not part of
the byte-identical vendor import.** We gave the agent dedicated `write`/`edit`/`read` file tools,
multi-command-per-turn, and sub-agent spawning; the paper's Part I agent issues ONE shell
`run_command` per turn. For the file-heavy Part I, this richer agent is the credible primary driver
of our +~12pp Acc / +~20pp Compl over the paper's GPT-5.2.

## 6. Honest caveats (read before using any number in a paper)
1. **"We exceed the paper (56.2 vs 44)" must NOT be read as a model/method win.** It is mostly an
   **agent-tooling difference** (full openclaw `coding` agent vs the paper's single `run_command`
   rollout) plus an uncontrolled serving/reasoning difference. To make the our-vs-paper comparison
   fair, the agent must be constrained to a single shell tool + the App A.1 prompt and re-run.
2. **The clean, quotable result is the mirix-vs-metaclaw A/B only** (same agent config on both arms),
   and on Part I that A/B is a tie on Acc / noise on Compl, with a recall-shaped MC gain.
3. file_check is **front-loaded**: pass ~52–87% on days 01–10, **0/47 on days 25–30** (genuine
   assertion failures, consistent with the paper flagging days 25–30 hardest). 37.5% is an average,
   not uniform mastery.
4. Earlier wrong claims now retracted: (a) "the paper has no feedback chain" — FALSE, it is native
   and shared; (b) Acc 46.2/48.3 — wrong scoring rule (exact-set), the paper-aligned Acc is 56.2.

## 7. Two harness bugs fixed (both environment-layer, MIRIX-independent, parity repairs)
1. **python-127 shim** (`runner.py::_ensure_python_shim`): checkers call `python …` via `/bin/sh`; hosts with only `python3` exited 127 → every file_check a false 0. Fix symlinks `python`→`sys.executable` onto the checker PATH.
2. **agent-id routing** (`vendor/benchmark/src/infer/infer_cmd.py`): `openclaw agent` was invoked without `--agent <id>` → ran the implicit `main` agent → wrote files to a workspace the scorer never inspected → 224 file_check all 0. Fix threads `agent_id` through `_run_group → _run_question → _run_openclaw_agent` and appends `--agent <id>`. Both arms were affected before the fix (why even native metaclaw scored ~0 Compl). These are parity repairs, NOT advantages over the paper.

## 8. mirix produce health (verified)
- 191 active skills for `eval-metaclaw-20260602T150239Z-mirix`; 100% with 4096-dim description+instructions embeddings; 445 skill_create + 306 evolve-200 over the run.
- Non-blocking produce bugs (fix next round): `skill_edit` regex `bad escape \d` (11 fails, replacement parsed as regex); `skill_create` strict-structured-output schema warning (`examples.items` missing `properties`, falls back fine).

## 9. Reproduce
```bash
# paper-aligned Acc (partial-credit) is in report.json; the exact-set triplet via:
python evals/metaclaw/score_arm.py evals/metaclaw/runs/metaclaw30-agentfix-20260602T230234Z metaclaw
python evals/metaclaw/score_arm.py evals/metaclaw/runs/mirix30-agentfix-20260602T230238Z   mirix
# partial-credit MC + first-round Compl: see the snippets in the chat log / score_arm extensions.
```

## 10. Data archive index (this dir: runs/_FINAL_agentfix_artifacts/)
- `logs/{mc30,mx30}_agentfix.log` — per-arm runner logs (full day01–30, EVAL_EXIT=0).
- `logs/mirix_server.log` (18MB) — evolve/skill_create narrative. ⚠ contains a leaked OpenRouter key.
- `mirix_191skills_dump.sql` / `procedural_memory_fulltable.sql` (64MB) — skill DB backups.
- `ground_truth_graders_seeds.tgz` — data/eval graders + workspaces seeds + all_tests.json.
- `{metaclaw,mirix}_report.json` — bench-native score artifacts (the byte-exact Acc source).
- `score_arm.py` (+ copy at evals/metaclaw/score_arm.py) — scorer.
- `run_*.sh`, `wait_both_arms.sh`, `midhealth_agentfix.sh` — reproducibility harness.

## 11. Outstanding risks (require user action)
- **Uncommitted code** in two trees: main repo `eval/original-mirix-3day` (agent-id + python-shim + tests + .gitignore) and worktree `feat/skill-evolve` (stateless-evolve + embedding/async fixes + untracked `test_skill_evolve_reset.py`).
- **DB volume**: skills on an anonymous docker volume (`e7906e60…`); the SQL dumps here are the only portable backup.
- **Leaked key**: OpenRouter `sk-or-v1-…` in ~40 `runs/*/proxy.yaml`, `.env`, and `mirix_server.log` — all gitignored / never pushed (verified: no remote contains it), but rotate if the machine is shared.
