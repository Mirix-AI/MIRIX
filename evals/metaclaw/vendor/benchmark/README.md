# MetaClaw Benchmark

Evaluation suite for the MetaClaw Evolution Benchmark — measures how well AI
agents learn and adapt from multi-day interaction histories.

## Quick Start

```bash
# Install (requires Python ≥ 3.10)
cd benchmark
pip install -e .

# Set required environment variables (see docs/scripts.md for the full list)
export BENCHMARK_BASE_URL=http://127.0.0.1:30000/v1   # OpenAI-compatible API base URL
export BENCHMARK_API_KEY=<your-api-key>                # API key for the above endpoint
export BENCHMARK_MODEL=<your-model-id>                 # Model ID expected by the API server

# Validate the dataset (run from the project root: MetaClaw/)
metaclaw-bench check -p benchmark/data/metaclaw-bench/all_tests.json

# Run the full pipeline (infer → scoring → report)
metaclaw-bench run \
  -i benchmark/data/metaclaw-bench/all_tests.json \
  -o benchmark/results
```

`BENCHMARK_MODEL` is the model ID string that your API server uses (e.g.
`gpt-5.2`, `Kimi-K2.5`).  It is injected into the openclaw agent config as the
primary model.

For ready-made experiment runner scripts (with proxy, skills, memory, and RL
support) see **[docs/scripts.md](docs/scripts.md)**.

## Using the Experiment Scripts

The scripts under `benchmark/scripts/` handle proxy lifecycle, port allocation,
and logging automatically.

**Recommended: always set `METACLAW_ROOT` to an absolute path** before running
any script.  Relative paths are resolved from the current working directory and
may silently fail when the script is called from an unexpected location.  The
same applies to `METACLAW_SKILLS_DIR` and `METACLAW_API_KEY_SCRIPT`.

**Concurrency:** scripts that activate MetaClaw features (memory, skills, RL)
**must run serially** — each test day depends on the state accumulated in
previous days.  These scripts hardcode `-w 1`.  `baseline_run.py` and
`proxy_passthrough_run.py` carry no cross-day state and default to `-w 15`
(parallel) — different scenes are independent, only rounds within a scene are
sequential.

<details>
<summary><strong>Example: baseline (no proxy)</strong></summary>

```bash
# 1. Required LLM credentials
export BENCHMARK_BASE_URL=https://api.openai.com/v1
export BENCHMARK_API_KEY=sk-...
export BENCHMARK_MODEL=gpt-5.2           # model ID expected by your API server

# 2. Strongly recommended: pin the project root to an absolute path
export METACLAW_ROOT=/absolute/path/to/MetaClaw

# 3. Run
python /absolute/path/to/MetaClaw/benchmark/scripts/baseline_run.py
```

</details>

<details>
<summary><strong>Example: memory mode</strong></summary>

```bash
export BENCHMARK_BASE_URL=https://api.openai.com/v1
export BENCHMARK_API_KEY=sk-...
export BENCHMARK_MODEL=gpt-5.2
export METACLAW_ROOT=/absolute/path/to/MetaClaw

# memory_run.py starts a proxy — no extra vars needed for memory-only mode
python /absolute/path/to/MetaClaw/benchmark/scripts/memory_run.py
```

</details>

<details>
<summary><strong>Example: skills mode</strong></summary>

```bash
export BENCHMARK_BASE_URL=https://api.openai.com/v1
export BENCHMARK_API_KEY=sk-...
export BENCHMARK_MODEL=gpt-5.2
export METACLAW_ROOT=/absolute/path/to/MetaClaw
export METACLAW_SKILLS_DIR=/absolute/path/to/your/skills   # required for skills scripts

python /absolute/path/to/MetaClaw/benchmark/scripts/skills_only_run.py
```

</details>

<details>
<summary><strong>Example: RL mode</strong></summary>

```bash
# Skill evolver uses BENCHMARK_* vars
# export BENCHMARK_BASE_URL=https://api.openai.com/v1
# export BENCHMARK_API_KEY=sk-...
# export BENCHMARK_MODEL=gpt-5.2
# export METACLAW_ROOT=/absolute/path/to/MetaClaw

# Additional vars for RL training
export TINKER_KEY=<tinker-api-key>
export TINKER_MODEL=<model-id-for-rl>
export PRM_MODEL=<process-reward-model-id>

python /absolute/path/to/MetaClaw/benchmark/scripts/rl_only_run.py
```

</details>

For a ready-to-fill template copy `benchmark/scripts/_env_arg_example.sh`.
Full variable reference: **[docs/env.md](docs/env.md)**.
Per-script breakdown: **[docs/scripts.md](docs/scripts.md)**.

## Project Structure

```
benchmark/
├── data/
│   ├── metaclaw-bench/          # Full benchmark (30 days)
│   └── metaclaw-bench-small/    # Small subset (12 days)
├── docs/
│   ├── CLI.md                   # CLI reference (all commands and flags)
│   ├── scripts.md               # Experiment runner scripts guide
│   └── env.md                   # Environment variable reference
├── scripts/                     # Experiment runner scripts
│   └── config/                  # YAML configs for each strategy
│   └── _env_arg_example.sh      # Environment variable template
├── src/                         # Core library
│   ├── cli.py                   # Entry point
│   ├── check/                   # Dataset validation
│   ├── infer/                   # Agent inference
│   ├── scoring/                 # Result scoring
│   ├── report/                  # Report generation
│   ├── run/                     # Full pipeline orchestration
│   └── clean/                   # Workspace cleanup
├── tests/                       # Unit tests
└── openclaw_customize/          # OpenClaw plugin extensions
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `check` | Validate dataset integrity (8 checks) |
| `infer` | Run agent inference on scenarios |
| `scoring` | Score inference results |
| `report` | Generate summary report |
| `report-ratio` | Compute compaction ratios between reports |
| `run` | Full pipeline: infer → scoring → report |
| `clean` | Remove temporary work directories |

See **[docs/CLI.md](docs/CLI.md)** for full usage, all flags, and examples.

## Experiment Scripts

Pre-built runner scripts under `scripts/` support various agent strategies:

| Script | Features |
|--------|----------|
| `baseline_run.py` | No proxy — direct API calls |
| `proxy_passthrough_run.py` | Proxy in passthrough mode (no enhancements) |
| `skills_only_run.py` | Proxy with pre-built skills |
| `memory_run.py` | Proxy with memory extraction/injection |
| `rl_only_run.py` | Proxy with RL training |
| `rl_run.py` | RL + skills |
| `skills_memory_run.py` | Skills + memory |
| `rl_only_memory_run.py` | RL + memory |
| `madmax_memory_run.py` | RL + skills + memory (all features) |

Each script reads environment variables from the shell; no path editing is
needed.  See **[docs/scripts.md](docs/scripts.md)** for setup instructions,
environment variable reference, and a per-script description.

## Development

```bash
pip install -e ".[dev]"
pytest -v tests/
```
