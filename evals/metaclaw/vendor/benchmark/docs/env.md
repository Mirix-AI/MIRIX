# Environment Variable Reference

This document lists every environment variable recognized by the MetaClaw
benchmark framework, grouped by who sets them and where they are consumed.

---

## User-Facing Variables

These are set by the user (or by sourcing `scripts/_env_arg_example.sh`)
before running any script or CLI command.

### LLM API credentials

| Variable | Required | Description |
|----------|----------|-------------|
| `BENCHMARK_BASE_URL` | yes | Base URL of the OpenAI-compatible API endpoint, e.g. `https://api.openai.com/v1` |
| `BENCHMARK_API_KEY` | yes | API key for the above endpoint |
| `BENCHMARK_MODEL` | yes | Model ID as expected by the API server, e.g. `gpt-4o` |

**Where consumed:**

- `data/*/openclaw_cfg/openclaw.json` — injected as `"baseUrl"`, `"apiKey"`, model `"id"`, and `"primary"` key via `${BENCHMARK_BASE_URL}`, `${BENCHMARK_API_KEY}`, `${BENCHMARK_MODEL}` placeholders.  `infer_cmd.py` rewrites these placeholders into the per-test work copy of `openclaw.json` at runtime.
- `scripts/config/*.yaml` — injected as `llm.api_base`, `llm.api_key`, `llm.model_id` when the script calls `write_proxy_config()`.

> `BENCHMARK_MODEL` must exactly match the model ID your API server expects.
> It is embedded in the agent config as `"primary": "metaclaw-bench/${BENCHMARK_MODEL}"`.

---

### Project root

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `METACLAW_ROOT` | no | auto-detected | Absolute path to the MetaClaw project root (the directory that contains `benchmark/`) |

**Where consumed:**

- `src/utils.py:get_project_root()` — all relative path resolution in the CLI goes through this function.  If unset, it falls back to `Path(__file__).parent.parent.parent` (three levels above `src/utils.py`), which works correctly when the package is installed in-place from `benchmark/`.
- `src/infer/infer_cmd.py` — passed as `METACLAW_ROOT` in the environment of every openclaw gateway and agent subprocess so that `${METACLAW_ROOT}` placeholders in `openclaw.json` are resolved correctly.
- `data/*/openclaw_cfg/openclaw.json` — referenced in `"agentDir"` and the plugin `"logDir"` config as `${METACLAW_ROOT}/...`.
- All `scripts/*.py` — `_METACLAW_ROOT = Path(os.environ.get("METACLAW_ROOT") or _SCRIPT_DIR.parent.parent)` to derive default paths for logs, inputs, and outputs.

**Recommendation:** always set this to an absolute path to avoid working-directory-dependent failures.

---

### Binary overrides

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `METACLAW_BENCH_BIN` | no | `metaclaw-bench` | Explicit path to the `metaclaw-bench` binary |
| `METACLAW_BIN` | no | `metaclaw` | Explicit path to the `metaclaw` binary (used by `proxy_run.py`) |

**Where consumed:**

- `METACLAW_BENCH_BIN` — `cfg.BENCH_BIN` in all `scripts/*.py` (except `proxy_run.py`).  Defaults to `"metaclaw-bench"`, which is resolved via `PATH`.
- `METACLAW_BIN` — `cfg.METACLAW_BIN` in `scripts/proxy_run.py`.  Defaults to `"metaclaw"`, resolved via `PATH`.

Set these when the binaries are not on `PATH` or when you need to pin a specific installation.

---

### Credential loader script

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `METACLAW_API_KEY_SCRIPT` | no | unset | Absolute path to a shell script that exports credentials.  The script is `source`d before the benchmark runs. |

**Where consumed:**

- `cfg.API_KEY_SCRIPT` in all `scripts/*.py`.  When set, the script calls `load_env_from_shell(cfg.API_KEY_SCRIPT)` to load the environment before constructing any subprocesses.  If unset, the current shell environment is used as-is.

Useful when credentials are managed by a secrets manager or a separate auth helper.

---

### Benchmark input file

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `METACLAW_BENCH_INPUT` | no | per-script default | Absolute path to the `all_tests*.json` file used as benchmark input.  Overrides the built-in default in every runner script simultaneously.  Useful for switching between the full dataset and a smaller subset (e.g. `metaclaw-bench-small`) without modifying any script. |

**Default values by script:**

- `baseline_run.py` — `<METACLAW_ROOT>/benchmark/data/metaclaw-bench/all_tests.json`
- all other scripts — `<METACLAW_ROOT>/benchmark/data/metaclaw-bench/all_tests_metaclaw.json`

**Where consumed:**

- `cfg.BENCH_INPUT` in all `scripts/*.py` — resolved at import time via `os.environ.get("METACLAW_BENCH_INPUT", <default>)`.

---

### Skills directory

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `METACLAW_SKILLS_DIR` | required for skills scripts | `<METACLAW_ROOT>/memory_data/skills` | Path to the directory containing pre-built skills |

**Where consumed:**

- `cfg.ORIGINAL_SKILL_DIR` in `scripts/proxy_passthrough_run.py`, `skills_only_run.py`, `rl_run.py`, `skills_memory_run.py`, `madmax_memory_run.py`.
- Each script copies this directory to a temporary location before starting the proxy so that every run begins from the same initial skill state.
- The temporary copy path is then written into `skills.dir` inside the proxy YAML config via `write_proxy_config()`.

---

### RL-specific variables

Required only for scripts that use RL training (`rl_only_run.py`, `rl_run.py`,
`rl_only_memory_run.py`, `madmax_memory_run.py`).

| Variable | Description |
|----------|-------------|
| `TINKER_KEY` | API key for the Tinker RL fine-tuning service |
| `TINKER_MODEL` | Model ID used by the RL fine-tuning step |
| `PRM_MODEL` | Process Reward Model ID used for RL scoring |

**Where consumed:**

- `scripts/config/rl.yaml`, `rl-only.yaml`, `rl-only-memory.yaml`, `madmax.yaml` — injected as `rl.model`, `rl.tinker_api_key`, `rl.prm_model`, `rl.prm_api_key` via `${VAR}` placeholders expanded in `write_proxy_config()`.

---

## Internal Variables

These are set **by the framework** at runtime and are not intended to be
configured by the user directly.  They are documented here for reference when
debugging or extending the benchmark.

### `METACLAW_PROXY_PORT`

Set by runner scripts (`scripts/*.py`) after calling `find_free_port()`.
Injected into the `metaclaw-bench` subprocess environment so that
`${METACLAW_PROXY_PORT}` placeholders in `openclaw.json` resolve to the
dynamically chosen port.

**Consumed in `src/infer/infer_cmd.py`:**

- Line ~143: `proxy_port = os.environ.get("METACLAW_PROXY_PORT", "30000")` — replaces `${METACLAW_PROXY_PORT}` in the work copy of `openclaw.json`.
- Line ~1129: read again in `_trigger_train_step()` to pass `--port` to `metaclaw train-step`.

---

### `BENCHMARK_WORKSPACE_DIR`

A placeholder in `data/*/openclaw_cfg/openclaw.json`:

```json
"workspace": "${BENCHMARK_WORKSPACE_DIR}"
```

This placeholder is **never expanded via the environment**.  Instead,
`infer_cmd.py:_patch_agent_workspace()` directly rewrites the agent's
`"workspace"` field in the per-test work copy of `openclaw.json` with the
actual workspace path for that test.  The placeholder just marks the field as
needing per-test injection.

---

### `OPENCLAW_CONFIG_PATH`, `OPENCLAW_STATE_DIR`, `OPENCLAW_GATEWAY_PORT`

Set by `src/infer/infer_cmd.py` when spawning `openclaw gateway run` and
`openclaw agent` subprocesses.  Each test gets a fully isolated work copy of
the openclaw state and a dedicated gateway process on a free port.

| Variable | Set in | Passed to |
|----------|--------|-----------|
| `OPENCLAW_CONFIG_PATH` | `_start_work_gateway()`, `_run_openclaw_agent()` | `openclaw gateway run`, `openclaw agent` |
| `OPENCLAW_STATE_DIR` | `_start_work_gateway()`, `_run_openclaw_agent()` | `openclaw gateway run`, `openclaw agent` |
| `OPENCLAW_GATEWAY_PORT` | `_start_work_gateway()`, `_run_openclaw_agent()` | `openclaw gateway run`, `openclaw agent` |

---

### `METACLAW_CONFIG_FILE`

Set by runner scripts (`scripts/*.py`) before launching `proxy_run.py` as a
subprocess:

```python
proxy_env["METACLAW_CONFIG_FILE"] = config_path
```

`proxy_run.py` reads this variable and passes `--config <path>` to
`metaclaw start`, pointing it at the dynamically generated (port-patched) YAML
config for that run.
