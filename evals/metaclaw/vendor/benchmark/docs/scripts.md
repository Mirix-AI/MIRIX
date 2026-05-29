# Experiment Runner Scripts

Pre-built runner scripts under `benchmark/scripts/` automate the full
experiment workflow: setting up a MetaClaw proxy, running the benchmark, and
collecting results.  Each script is self-contained and can be run directly
with `python scripts/<script>.py` from the project root.

---

## Environment Setup

All scripts read environment variables for credentials and paths.  Copy and
fill in `scripts/_env_arg_example.sh`, then source it before running:

```bash
cp benchmark/scripts/_env_arg_example.sh ~/.metaclaw_bench_env.sh
# Edit ~/.metaclaw_bench_env.sh with your values
source ~/.metaclaw_bench_env.sh
```

### Required Variables

| Variable | Description |
|----------|-------------|
| `BENCHMARK_BASE_URL` | Base URL of the OpenAI-compatible API endpoint (e.g. `https://api.openai.com/v1`) |
| `BENCHMARK_API_KEY` | API key for the above endpoint |
| `BENCHMARK_MODEL` | Model ID as registered in your API server; used in `openclaw.json` as the agent model |

> `BENCHMARK_MODEL` must exactly match the model ID your API server expects.
> It is injected into the openclaw config as `"primary": "metaclaw-bench/${BENCHMARK_MODEL}"`.

### RL-only Variables (required for RL scripts)

| Variable | Description |
|----------|-------------|
| `TINKER_KEY` | API key for the Tinker (RL training) service |
| `TINKER_MODEL` | Model ID used by the RL fine-tuning step |
| `PRM_MODEL` | Process Reward Model ID used for RL scoring |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `METACLAW_ROOT` | Auto-detected from script location | Absolute path to the MetaClaw project root |
| `METACLAW_BENCH_BIN` | `metaclaw-bench` (resolved via `PATH`) | Explicit path to the `metaclaw-bench` binary |
| `METACLAW_BIN` | `metaclaw` (resolved via `PATH`) | Explicit path to the `metaclaw` binary (`proxy_run.py`) |
| `METACLAW_API_KEY_SCRIPT` | unset (skip) | Shell script to `source` for exporting env vars; useful when credentials are managed externally |
| `METACLAW_SKILLS_DIR` | `<METACLAW_ROOT>/memory_data/skills` | Directory containing pre-built skills (required for skills-based scripts) |
| `METACLAW_BENCH_INPUT` | per-script default (see table below) | Override the benchmark input file for all scripts; useful for switching to a smaller dataset without editing individual scripts |

---

## Script Reference

All scripts auto-detect paths from `METACLAW_ROOT` (or the script's own
location when `METACLAW_ROOT` is unset).  No path editing is needed unless
you want to override defaults.

### `baseline_run.py` — No proxy

Runs the benchmark **without** the MetaClaw proxy.  The agent communicates
directly with `BENCHMARK_BASE_URL`.  Use this to establish the baseline score.

```bash
python benchmark/scripts/baseline_run.py
```

Config: none (no proxy launched).  
Input: `all_tests.json` (standard, without metaclaw agent config).

---

### `proxy_passthrough_run.py` — Proxy in passthrough mode

Starts the MetaClaw proxy with **all enhancements disabled** (no skills, no
memory, no RL) and routes agent requests through it.  Useful for validating the
proxy plumbing or measuring its raw overhead without any feature active.

```bash
python benchmark/scripts/proxy_passthrough_run.py
```

Config: `scripts/config/dummy.yaml`.  
Requires: `METACLAW_SKILLS_DIR` (needed by the proxy config even with skills off).

---

### `skills_only_run.py` — Skills only

Runs through the proxy with pre-built skills injected into the agent context.
No memory or RL.

```bash
python benchmark/scripts/skills_only_run.py
```

Config: `scripts/config/skills-only.yaml`.  
Requires: `METACLAW_SKILLS_DIR`.

---

### `memory_run.py` — Memory only

Runs through the proxy with memory extraction and injection active.  Generates
`memory_report.md` alongside the regular benchmark report.

```bash
python benchmark/scripts/memory_run.py
```

Config: `scripts/config/memory.yaml`.  
Memory store: `benchmark/memory_runs/<timestamp>/`.

---

### `rl_only_run.py` — RL only

Runs through the proxy with RL training triggered every `SCENE_PER_TRAIN`
completed test scenes.  No skills or memory.

```bash
python benchmark/scripts/rl_only_run.py
```

Config: `scripts/config/rl-only.yaml`.  
Requires RL variables: `TINKER_KEY`, `TINKER_MODEL`, `PRM_MODEL`.

---

### `rl_run.py` — RL + skills

Runs through the proxy with both RL training and pre-built skills active.

```bash
python benchmark/scripts/rl_run.py
```

Config: `scripts/config/rl.yaml`.  
Requires: `METACLAW_SKILLS_DIR` + RL variables.

---

### `skills_memory_run.py` — Skills + memory

Combines skills and memory; no RL.

```bash
python benchmark/scripts/skills_memory_run.py
```

Config: `scripts/config/skills-memory.yaml`.  
Requires: `METACLAW_SKILLS_DIR`.

---

### `rl_only_memory_run.py` — RL + memory (no skills)

Combines RL training and memory; no skills.

```bash
python benchmark/scripts/rl_only_memory_run.py
```

Config: `scripts/config/rl-only-memory.yaml`.  
Requires RL variables.

---

### `madmax_memory_run.py` — RL + skills + memory

All features enabled simultaneously.

```bash
python benchmark/scripts/madmax_memory_run.py
```

Config: `scripts/config/madmax.yaml`.  
Requires: `METACLAW_SKILLS_DIR` + RL variables.

---

### `proxy_run.py` — Standalone proxy launcher

Starts `metaclaw start` and tees output to both terminal and log.  Invoked
automatically by the other runner scripts; can also be run directly when you
need a standalone proxy.

```bash
python benchmark/scripts/proxy_run.py
```

Pass the proxy config via `METACLAW_CONFIG_FILE` before launching:

```bash
METACLAW_CONFIG_FILE=/path/to/config.yaml python benchmark/scripts/proxy_run.py
```

---

## Script Comparison

| Script | Proxy | Skills | Memory | RL | Workers | Default input file |
|--------|-------|--------|--------|----|---------|-------------------|
| `baseline_run.py` | — | — | — | — | parallel (`-w 15`) | `all_tests.json` |
| `proxy_passthrough_run.py` | ✓ | — | — | — | parallel (`-w 15`) | `all_tests_metaclaw.json` |
| `skills_only_run.py` | ✓ | ✓ | — | — | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `memory_run.py` | ✓ | — | ✓ | — | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `buffer_memory_run.py` | ✓ | — | ✓ (incremental) | — | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `rl_only_run.py` | ✓ | — | — | ✓ | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `rl_run.py` | ✓ | ✓ | — | ✓ | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `skills_memory_run.py` | ✓ | ✓ | ✓ | — | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `rl_only_memory_run.py` | ✓ | — | ✓ | ✓ | serial (`-w 1`) | `all_tests_metaclaw.json` |
| `madmax_memory_run.py` | ✓ | ✓ | ✓ | ✓ | serial (`-w 1`) | `all_tests_metaclaw.json` |

All default paths resolve under `<METACLAW_ROOT>/benchmark/data/metaclaw-bench/`.
Set `METACLAW_BENCH_INPUT` to an absolute path to override for every script at once,
e.g. to point at the smaller `metaclaw-bench-small` dataset.

**Why the serial constraint?**  Any script that activates MetaClaw state (memory extracts, skill updates, RL checkpoints) relies on strict ordering: day N+1 must see the state accumulated through day N.  Running tests concurrently would corrupt the accumulated state.

`baseline_run.py` and `proxy_passthrough_run.py` have no such dependency — for a plain LLM inference engine the rounds *within* a single scene are sequential, but different scenes (days) are fully independent, so parallel execution is safe.

---

## YAML Config Files

Each YAML file under `scripts/config/` controls the MetaClaw proxy for a
specific experiment variant.  `${VAR}` placeholders are expanded from the
current environment at runtime.  See the table above to match scripts to
their config files.

The `env_arg_example.sh` file contains a ready-to-fill template for all
required environment variables.

---

## Logs

All scripts write a run log to:

```
benchmark/logs/<script_name>/bench_run.log
```

If the file already exists a numeric suffix is appended (`_1`, `_2`, …) so
previous logs are never overwritten.

---

## See Also

- **[CLI.md](CLI.md)** — `metaclaw-bench` command reference
- **[env.md](env.md)** — complete environment variable reference (user-facing and internal)
