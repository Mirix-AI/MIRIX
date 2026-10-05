# skill_bench: usage and reproduction

## 1. What this is

`evals/skill_bench/` is the evaluation harness for MIRIX's skill learning. It measures how an agent does on held-out tasks after a memory method has learned from earlier tasks.

The study compares four memory methods:
- `none`: no memory;
- `reasoningbank`: ReasoningBank;
- `awm`: Agent Workflow Memory;
- `mirix`: MIRIX procedural memory.

They run on two benchmarks, ALFWorld and WebShop, with two models through OpenRouter: `luna` = `openai/gpt-6-luna` and `luna56` = `openai/gpt-5.6-luna`.

Each **cell** (benchmark × model × method) runs in three stages:
1. it learns on the training split;
2. its memory is frozen;
3. it is tested once on the held-out test split.

The study, `--study phase16`, has 2 × 2 × 4 = 16 cells. It runs in three replicate runs directories, r1 to r3.

This release contains code only: no run records and no results. For how MIRIX's skill learning and the harness work inside, see [ARCHITECTURE.md](ARCHITECTURE.md).

All commands below run from the root of your checkout, written `<checkout>`, with the virtual environment at `<checkout>/.venv`.

## 2. Requirements

| Requirement | Why |
|---|---|
| **x86_64 Linux** | The WebShop server image is built for `linux/amd64` only (`envs/webshop.py`, `build_argv`; the Dockerfile sets an amd64 Java path). macOS can run ALFWorld cells only. |
| **Docker Engine with the Compose plugin** (`docker compose`) | Runs Postgres and the WebShop server. The user who runs the harness must be able to use `docker` without `sudo`: MIRIX checkpoints run `pg_dump` inside the Postgres container with `docker exec`. |
| **Python 3.11** | The harness and MIRIX. `pyproject.toml` allows 3.10 or newer; the constraints file `evals/skill_bench/requirements-lock.txt` was resolved with Python 3.11.13. |
| **uv** (recommended), or pip | Installs the environment from the constraints file (§3) |
| **git** | Cloning; every attempt records the commit it ran |
| **tmux** | `tools/phase16.sh up` starts each replicate in its own tmux session. Without tmux, use `phase16.sh exec` in the foreground (§8.1). |
| **An OpenRouter API key** | Preferably a key used for nothing else. The harness's budgets count **all** spend on the key since the budget started, whoever spent it (`--budget` in `infra/gateway.py`). |
| **Network access** | OpenRouter during runs. For setup: PyPI, GitHub (the ALFWorld data release, the WebShop source), Hugging Face (the WebShop product data), Docker Hub (the Postgres and Python base images) and `snapshot.debian.org` (Java packages for the WebShop image). |

**Memory.** These figures come from constants in the code:

| Consumer | Constant | Value |
|---|---|---|
| WebShop server container | `WEBSHOP_SERVER` in the study constants module (`evals/skill_bench/protocol.py`) | Capped at `--memory 24g`, `--cpus 2` |
| One cell (worker plus its MIRIX server), as estimated by the launch gate | `CELL_MEMORY_GB` in `infra/resources.py` | See the table below |
| Postgres | `infra/compose.yaml` | `shared_buffers=512MB`, `max_connections=2000` |
| Launch gate of `tools/phase16.sh` (`P16_GATES`) | `--min-free-memory-gb 8`, `--memory-emergency-gb 3` | A cell starts only if 8 GB stay available after its estimate. Below 3 GB, the newest worker is stopped. |

Estimated GB per cell (`CELL_MEMORY_GB`), by benchmark and method (RB = `reasoningbank`, AWM = `awm`, MIRIX = `mirix`):

| | none | RB | AWM | MIRIX |
|---|---|---|---|---|
| ALFWorld | 0.5 | 0.6 | 0.6 | 1.4 |
| WebShop | 0.35 | 0.45 | 0.5 | 1.4 |

The launcher runs up to 4 cells per replicate (`P16_CONCURRENCY`), so up to 12 cells at once, of which at most 6 are WebShop cells. Starting all 12 at once, all of them MIRIX cells, therefore needs about 12 × 1.4 GB + 24 GB (WebShop cap) + 8 GB (gate) ≈ 49 GB of available memory, plus the gateways, Postgres and the operating system. With less memory, the gate simply starts fewer cells at a time. The 24 GB figure is the WebShop container's cap; how much of it the server uses is not set by the code.

**Disk.** The launch gate keeps `--min-free-disk-gb 15` free under the runs directory. Space is used by:
- the ALFWorld game files (`$ALFWORLD_DATA`);
- the WebShop image: about 5.7 GB of pinned product data (the file sizes in `WEBSHOP_DATA`), its Lucene index, Java and Python layers, and Docker's build cache;
- the Postgres volume, with one database per MIRIX cell attempt;
- the runs directories: trajectories, gateway ledgers, the latest two learning checkpoints per cell (a full database dump for MIRIX cells), and what learning resumes rolled back.

The code sets no total for these; leave room beyond the gate's 15 GB.

## 3. Install

```bash
git clone <repository-url> <checkout>
cd <checkout>
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[skill-bench]' -c evals/skill_bench/requirements-lock.txt
```

With plain pip instead of uv:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[skill-bench]' -c evals/skill_bench/requirements-lock.txt
```

The `skill-bench` extra (`pyproject.toml`) adds ALFWorld, its game engine and the harness's other dependencies. The constraints file `evals/skill_bench/requirements-lock.txt` pins the packages of the environment to exact versions, resolved with Python 3.11.13. Among them are `alfworld==0.4.2` and `textworld==1.7.0`; the ALFWorld environment refuses any other installed version of the engine. The file lists one version per package and has no platform markers. Install with it, so that your environment matches the one the harness was tested with.

**Tokenizer vocabulary.** The gateway, MIRIX and the tests count tokens with tiktoken's `o200k_base`, read from tiktoken's cache. Cache it once, in a directory that survives reboots:

```bash
mkdir -p "$HOME/.cache/tiktoken"
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

Set the same `TIKTOKEN_CACHE_DIR` in the launcher config (§5.2), so that the gateway and each cell's MIRIX server read the same cache. Without it, they fall back to tiktoken's cache in the system temp directory.

Without the cached vocabulary:
- every harness test is skipped, and the skip message gives the command above;
- the gateway falls back to estimating tokens from characters, and every cell refuses to run on it.

On a host without internet access, copy the cache directory from a machine that has fetched it, and point `TIKTOKEN_CACHE_DIR` at the copy.

## 4. Data and services

### 4.1 ALFWorld data

```bash
export ALFWORLD_DATA="$HOME/.cache/alfworld"   # the default if unset
.venv/bin/alfworld-download
```

`alfworld-download` writes to `$ALFWORLD_DATA`. Before playing a game, the harness checks the game files of every split it uses against `envs/splits/alfworld/gamefiles.sha256.json`, and it refuses a game whose files differ. The preflight check (§6) reports this as `data:alfworld`.

### 4.2 Postgres (MIRIX cells)

```bash
docker compose -f evals/skill_bench/infra/compose.yaml up -d
docker compose -f evals/skill_bench/infra/compose.yaml ps    # wait until the service is "healthy"
```

The service has these properties:
- **Image:** `pgvector/pgvector:pg16`, pinned by digest.
- **Port:** `127.0.0.1:${SKILL_BENCH_PG_PORT:-55432}`, loopback only.
- **Healthy** only after `infra/init.sql` has created the template database `skill_bench_template` with the `vector`, `pg_trgm` and `uuid-ossp` extensions. This runs once, when the data volume is first created.
- **Restart policy:** `unless-stopped`, so it comes back with the Docker daemon.

Two variables change its network settings:

| Variable | Default | When to set it |
|---|---|---|
| `SKILL_BENCH_PG_PORT` | `55432` | The port is taken. Set it in the shell before `docker compose up` **and** in the launcher config (§5.2): the harness reads it too (`PostgresAdmin` in `infra/mirix_server.py`). |
| `SKILL_BENCH_SUBNET` | a private `/24` set in `infra/compose.yaml` | Docker reports the subnet overlapping an existing network, or its address pools are exhausted. Choose a free private `/24`. |

No Redis or Kafka is needed: every cell's MIRIX server runs with Redis disabled and MIRIX's in-memory queue.

### 4.3 The WebShop server (x86_64 Linux)

```bash
# Optional: show the docker build and docker run commands the CLI uses
.venv/bin/python -m evals.skill_bench.envs.webshop --print-argv

# Build the image (long; downloads about 5.7 GB of product data and builds a Lucene index)
.venv/bin/python -m evals.skill_bench.envs.webshop --build

# Start the server and wait up to 30 minutes until it is ready
.venv/bin/python -m evals.skill_bench.envs.webshop --start --wait-s 1800

# Check readiness at any time (prints /ready as JSON; exit 0 when ready)
.venv/bin/python -m evals.skill_bench.envs.webshop --status
```

The image build:
- clones princeton-nlp/WebShop at a pinned commit;
- downloads the full product set from a Hugging Face mirror, pinned by revision and verified by size and sha256;
- builds the Lucene index with one thread;
- records all of this in the image's build record.

The image tag is `skill-bench-webshop:<digest>`, where the digest is computed from the files of `envs/templates/webshop_server/`. A checkout whose server files differ therefore needs its own build.

`--start` runs the container `skill-bench-webshop`, published on `127.0.0.1:18560`, with the restart policy `unless-stopped` and the committed split files mounted read-only. If the container is already running, it is left as it is. If it exists but has stopped, it is replaced.

While loading the product set and the index, the server answers `loading`. When it is done it answers `ready`, or `failed` with a list of problems: before it serves, it checks every goal of the committed split files, the price table, the data and the index record. Use `--stop` to remove the container.

`--start-prepare` and `--stop-prepare` start and remove a separate container that is used only to draw new WebShop split files (`tools/prepare_webshop.py`). The committed split files are final, so a reproduction does not need it.

The index record is written so that two builds from the same product files, upstream commit and packages produce the same bytes (`envs/templates/webshop_server/data_index.py`). Its sha256 is compared with `index_sha256` in `envs/splits/webshop/source.json`: a server whose build record differs from the committed one answers `failed`, and no WebShop cell starts on it.

If you change `--port`, set `SKILL_BENCH_WEBSHOP_URL` (e.g. `http://127.0.0.1:<port>`) for the harness. Note that `phase16.sh recover` always starts the server on the default port.

## 5. Configuration

### 5.1 The key file (outside the checkout)

Keep the OpenRouter key in a dotenv file **outside** the checkout, readable only by you. The gateway is the only process that reads it during a run (ARCHITECTURE.md §5).

```bash
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench"
mkdir -p "$CFG"
install -m 600 /dev/null "$CFG/openrouter.env"
"${EDITOR:-vi}" "$CFG/openrouter.env"
# The file holds one line:
# OPENROUTER_API_KEY=<your OpenRouter key>
```

Every launcher command finds this file through `SKILL_BENCH_ENV_FILE` and passes it as `--env-file`.

### 5.2 The launcher config file

`tools/phase16.sh` sources `${P16_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env}` with bash, if the file exists. Every variable in it is exported to the supervisor, the scheduler, the gateway and the workers, apart from credential-like names, which the workers never receive. The file holds the key file's path, never the key. The systemd unit (§8.1) runs the launcher without your shell's environment, so keep the file at its default path if you use the unit.

```bash
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench"     # as in 5.1
cat > "$CFG/launch.env" <<EOF
# Sourced by evals/skill_bench/tools/phase16.sh; every variable is exported.
SKILL_BENCH_ENV_FILE=$CFG/openrouter.env
# Required for r1-r3: the USD the three replicates may spend together on this key.
P16_BUDGET=
ALFWORLD_DATA=$HOME/.cache/alfworld
TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken
# Optional, see the table in 5.4:
# SKILL_BENCH_PG_PORT=55432
# P16_CONCURRENCY=4
# P16_NOTIFY_CMD=<a command to run on every alert>
EOF
chmod 600 "$CFG/launch.env"
"${EDITOR:-vi}" "$CFG/launch.env"   # fill in P16_BUDGET (§8.5)
```

To use these settings in your own shell, for example for preflight:

```bash
set -a; . "${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env"; set +a
```

### 5.3 No `.env` file in the checkout or above it

Importing MIRIX runs `load_dotenv()`, which searches for a `.env` file upward from `mirix/`. A `.env` file in `<checkout>/mirix`, in `<checkout>`, or in any directory above it would let provider keys reach MIRIX past the gateway. MIRIX cells therefore refuse to start if one exists (`dotenv_files_visible_to_mirix` in `infra/mirix_server.py`).

MIRIX's standalone development setup puts a `.env` at the checkout root. Remove it, or use a separate checkout for the harness. Check:

```bash
.venv/bin/python -c "from evals.skill_bench.infra.mirix_server import dotenv_files_visible_to_mirix as f; print(f() or 'no .env visible to MIRIX')"
unset MIRIX_API_KEY   # the MIRIX method refuses to run while it is set
```

### 5.4 Environment variable reference

| Variable | Read by | Default | Meaning |
|---|---|---|---|
| `SKILL_BENCH_ENV_FILE` | `tools/phase16.sh` | none (required) | Path of the key file; passed as `--env-file` |
| `OPENROUTER_API_KEY` | `infra/gateway.py` (`load_api_key`), `tools/preflight.py` | none | Put it **in the key file**. It is read from the environment only when no `--env-file` is given. |
| `P16_CONFIG` | `tools/phase16.sh` | `${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env` | The launcher config file |
| `P16_BUDGET` | `tools/phase16.sh` | none (required for r1–r3) | Total USD budget of r1–r3 together, counted on the key's usage from one shared start |
| `P16_CONCURRENCY` | `tools/phase16.sh` | `4` | `--concurrency` of `dev` and r1–r3: cells per replicate (1–24) |
| `P16_GATES` | `tools/phase16.sh` | `--min-free-memory-gb 8 --min-free-disk-gb 15 --memory-emergency-gb 3` | The scheduler's resource gates |
| `P16_PORT_BASE` | `tools/phase16.sh` | `18540` | Gateway ports: r1, r2 and r3 use base + 1, + 2 and + 3; `pilot` base + 4; `dev` base + 5 |
| `P16_NOTIFY_CMD` | `tools/phase16.sh` → `supervise.py --notify-cmd` | none | Command run on every alert. It is split with `shlex`, never run through a shell, and receives the alert text in `$SKILL_BENCH_ALERT`. |
| `P16_PYTHON` | `tools/phase16.sh` | `.venv/bin/python` in the checkout | Interpreter |
| `P16_RUNS` | `tools/phase16.sh` | `evals/skill_bench/runs` | Parent of the `phase16-<step>` runs directories |
| `P16_WEBSHOP_WAIT_S` | `tools/phase16.sh recover` | `1800` | How long `recover` waits for the WebShop server |
| `P16_PILOT_BUDGET`, `P16_DEV_BUDGET` | `tools/phase16.sh` | set in `phase16.sh` | Budgets of the optional `pilot` and `dev` steps |
| `P16_ALLOW_OVERLAP` | `tools/phase16.sh` | unset | `1` starts a step while a step with another budget runs. Both then count each other's spend. |
| `P16_METHODS`, `P16_BENCHMARKS` | `tools/phase16.sh` | unset (the whole study) | Narrow r1–r3 to some methods or benchmarks. Leave them unset to reproduce the study. A later launch that selects more cells than an earlier one in the same runs directory is refused. |
| `ALFWORLD_DATA` | `envs/alfworld.py`, `alfworld-download`; exported by `phase16.sh` | `$HOME/.cache/alfworld` | ALFWorld game files |
| `TIKTOKEN_CACHE_DIR` | tiktoken, `infra/mirix_server.py` (`tiktoken_cache_dir`), the gateway, the tests | tiktoken's temp-directory cache | Where the `o200k_base` vocabulary is cached |
| `SKILL_BENCH_PG_PORT` | `infra/compose.yaml`, `infra/mirix_server.py` (`PostgresAdmin`) | `55432` | Host port of the compose Postgres |
| `SKILL_BENCH_SUBNET` | `infra/compose.yaml` | a private `/24` | Docker network subnet of the compose project |
| `SKILL_BENCH_PG_CONTAINER` | `infra/mirix_server.py` (`postgres_container`) | the container found by the compose labels | Postgres container that MIRIX checkpoints run `pg_dump` in |
| `SKILL_BENCH_WEBSHOP_URL` | `envs/webshop.py` | `http://127.0.0.1:18560` | WebShop server address |
| `SKILL_BENCH_HOST_DIR` | `host_registry.py` | `$HOME/.cache/skill_bench/host` | Directory that all schedulers of a host share. It must be the same for every replicate. |
| `SKILL_BENCH_ALERT` | set by `supervise.py` | — | Alert text passed to `--notify-cmd` |
| `MIRIX_API_KEY` | `methods/mirix.py` | must be unset | If set, the MIRIX client would authenticate as another tenant, so the method refuses to run |
| `SKILLOPT_DIR`, `VERL_AGENT_DIR`, `WEBSHOP_UPSTREAM_DIR` | tests only | unset (those tests skip) | Clones of SkillOpt at 79124b3, verl-agent at 20bd331 and WebShop at 64fa2a5, for the byte-level comparison tests of the prompts and environments |

Each cell's MIRIX server receives none of your `MIRIX_*` variables. It runs with MIRIX's defaults, plus fixed settings for retries, timeouts, Redis and the session idle timeout (`MIRIX_SERVER_ENV` in the study constants module).

## 6. Checks before spending

**1. Unit tests.** They need no model, key or network, but they do need the cached tokenizer vocabulary:

```bash
env -u OPENROUTER_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  .venv/bin/python -m pytest -m "not integration" evals/skill_bench/tests
```

**2. The study's cells.** This command writes nothing:

```bash
.venv/bin/python -m evals.skill_bench.schedule --study phase16 \
  --runs-dir evals/skill_bench/runs/phase16-r1 --dry-run
```

It prints 16 lines, one per cell: `alfworld-luna-mirix`, `alfworld-luna56-mirix`, … `webshop-luna56-none`.

**3. Preflight.** No model is called. The only requests that leave the machine read the key's balance from OpenRouter, which is free, and the models' public metadata. Preflight checks:
- the key file, the balance and the budget record of the runs directory;
- that `config/models.json` still matches OpenRouter's metadata for the models;
- the commit and code it would run, and the host;
- free memory and disk, against the launch gates;
- each benchmark's data and environment: the ALFWorld game files and engine versions, and a ready WebShop server built from this checkout with the committed split files;
- for MIRIX cells: no `.env` visible to MIRIX, no `MIRIX_API_KEY`, the compose Postgres with its template database, and MIRIX's retry and timeout settings.

```bash
set -a; . "${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env"; set +a
.venv/bin/python -m evals.skill_bench.tools.preflight \
  --env-file "$SKILL_BENCH_ENV_FILE" \
  --study phase16 \
  --runs-dir evals/skill_bench/runs/phase16-r1 \
  --budget "$P16_BUDGET" \
  --min-free-memory-gb 8 --min-free-disk-gb 15
```

Each check prints one line: `ok`, `warn`, `skip` or `FAIL`. Preflight must end with the line `preflight: ready (no check failed)` and exit status 0 before you launch. Warnings and skips do not block; a `warn` on the `code` line means tracked files differ from the commit, so commit or discard those changes first. The `balance` check requires the key's remaining balance to cover the gateway's pause floor (10 USD by default) plus `--budget`. `--offline` skips every network check.

## 7. A smoke run

A smoke cell learns on the first N training items and tests on the first N **validation** items, never on the test split. Its id ends in `-smoke-l<N>-t<N>`. Run smoke cells in their own runs directory, and never while the replicates run: every budget counts all spend on the key.

```bash
set -a; . "${XDG_CONFIG_HOME:-$HOME/.config}/skill-bench/launch.env"; set +a
SMOKE=evals/skill_bench/runs/smoke
SMOKE_BUDGET_USD=        # fill in: the most this smoke directory may spend

.venv/bin/python -m evals.skill_bench.run \
  --benchmark alfworld --model luna --method mirix \
  --split-mode smoke --n-train 2 --n-test 2 \
  --out "$SMOKE/alfworld-luna-mirix-smoke-l2-t2" \
  --gateway-runs-dir "$SMOKE" \
  --env-file "$SKILL_BENCH_ENV_FILE" \
  --budget "$SMOKE_BUDGET_USD"
```

Without `--gateway-url`, `run.py` launches its own gateway on a free port and stops it at the end. The gateway's ledger and budget record are written under `--gateway-runs-dir`. `run.py` prints one JSON line with the cell's status and exits with that status's code:

| Exit | Status | Next step |
|---|---|---|
| 0 | `complete` | Done |
| 1 | `crashed` | Run the same command again |
| 2 | `test_incomplete` | Run again: only the pending test items run |
| 3 | `learning_failed` | A harness failure stopped learning. Look at `attempt-NN/failures.jsonl`, then rerun with `--force-resume` (from the checkpoint) or `--new-attempt` (from scratch). |
| 4 | `refused` or `violation` | Read the message. The cell needs a person. |
| 5 | `busy` | Another `run.py` holds the cell |
| 6 | `budget_paused` | The gateway is paused (balance or budget) |
| 7 | `infra_unavailable` | Gateway, Postgres or the WebShop server was unreachable. Run again later. |
| 8 | `learning_interrupted` | Run again: learning resumes from the latest checkpoint |
| 130 | `interrupted` | Stopped by a signal. Run again to resume. |

Several smoke cells can run behind one gateway with the scheduler. Here is every method on both benchmarks with `luna`:

```bash
.venv/bin/python -m evals.skill_bench.schedule --runs-dir "$SMOKE" \
  --benchmark alfworld webshop --model luna \
  --method none reasoningbank awm mirix \
  --split-mode smoke --n-train 5 --n-test 5 \
  --concurrency 4 --gateway-port 18550 \
  --env-file "$SKILL_BENCH_ENV_FILE" --budget "$SMOKE_BUDGET_USD"
```

Inspect the smoke cells offline:

```bash
.venv/bin/python -m evals.skill_bench.audit  --runs-dir "$SMOKE" --split-mode smoke
.venv/bin/python -m evals.skill_bench.report --runs-dir "$SMOKE" --split-mode smoke
.venv/bin/python -m evals.skill_bench.tools.summarize_learning --runs-dir "$SMOKE" --split-mode smoke
```

`summarize_learning` prints what each method had learned when its memory was frozen: skills and learning runs for MIRIX, memory items for ReasoningBank, the workflow for AWM. It also prints the test-phase injections and each cell's cost from its ledgers.

## 8. Running the study

### 8.1 Start the three replicates

Run all three replicates from one unmodified checkout. Leave `P16_METHODS` and `P16_BENCHMARKS` unset, so each replicate runs all 16 cells.

```bash
cd <checkout>
evals/skill_bench/tools/phase16.sh up r1 r2 r3
evals/skill_bench/tools/phase16.sh status r1 r2 r3
tmux attach -t p16-r1          # watch a supervisor; detach with Ctrl-b d
```

`up` starts one supervisor per replicate, each in its own tmux session `p16-<step>`. A step whose supervisor is already running is left alone. To see the exact supervisor command line without starting anything:

```bash
evals/skill_bench/tools/phase16.sh print r1
```

The launcher's commands (`STEP` is `pilot`, `dev`, `r1`, `r2` or `r3`):

| Command | Effect |
|---|---|
| `print STEP [-- EXTRA...]` | Prints the supervisor's command line; changes nothing |
| `exec STEP [-- EXTRA...]` | Runs the supervisor in the foreground |
| `up STEP... [-- EXTRA...]` | Starts each step's supervisor in tmux session `p16-STEP`, unless a supervisor already holds its runs directory |
| `status [STEP...]` | `supervise.py --status` for each step (default: every step) |
| `recover STEP... [-- EXTRA...]` | Starts the compose Postgres and the WebShop server, then runs `up` (§9) |
| `install-unit` | Writes the systemd user unit for this checkout (below) |

The launcher reads its config (§5.2) on every command. Every command that acts on a step (`status` without steps acts on all five) needs `SKILL_BENCH_ENV_FILE`, and the replicate steps also need `P16_BUDGET`.

Each replicate's supervisor runs with these settings:

| Setting | r1 | r2 | r3 |
|---|---|---|---|
| Runs directory | `evals/skill_bench/runs/phase16-r1` | `…/phase16-r2` | `…/phase16-r3` |
| Gateway port (`P16_PORT_BASE` + N) | 18541 | 18542 | 18543 |
| Budget | `P16_BUDGET` | `P16_BUDGET`, counted from r1's start (`--budget-start-from …/phase16-r1`) | same as r2 |
| Fixed options | `--study phase16 --concurrency $P16_CONCURRENCY --test-concurrency 4 --balance-floor 10` and `P16_GATES` | same | same |

The three replicates share one budget start, so `P16_BUDGET` is their **total**. r2 and r3 wait, and are retried by their supervisors, until r1's gateway has recorded the start.

**Without tmux**, run each replicate in the foreground in its own terminal:

```bash
evals/skill_bench/tools/phase16.sh exec r1
```

**With systemd** (optional): `phase16.sh install-unit` fills this checkout's path into the template `evals/skill_bench/infra/skill-bench@.service.in` and writes the result to `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/skill-bench@.service`. It refuses a checkout path that holds a space, a quote, a backslash or `%`. It changes nothing else and prints the commands to run next:

```bash
evals/skill_bench/tools/phase16.sh install-unit
systemctl --user daemon-reload
systemctl --user enable --now skill-bench@r1 skill-bench@r2 skill-bench@r3
sudo loginctl enable-linger "$USER"     # start user units at boot, without a login
```

Each instance `skill-bench@STEP` runs `phase16.sh exec STEP`. Once enabled, it starts at boot and is restarted 60 s after a crash (`Restart=on-failure`), but not after an exit that needs a person (2, 10–13) or a stop you asked for (130, e.g. `systemctl --user stop skill-bench@r1`). Stopping the unit signals only the supervisor, which drains its workers itself.

Use either tmux or systemd for a step, not both. A second supervisor on the same runs directory exits with code 13.

### 8.2 How cells are scheduled and resumed

The process chain is: supervisor → `schedule.py --auto-resume` → one `run.py` worker per cell → a MIRIX server for MIRIX cells.

- **Order.** ALFWorld cells start before WebShop cells. Within a benchmark, `mirix`, `reasoningbank` and `awm` start before `none`. WebShop cells wait until the WebShop server answers `ready`.
- **Learning.** Training items run one at a time, in a fixed seeded order. After setup and after every item, a learning checkpoint is written. An interruption (a crash, a signal, a reboot, an outage) leaves the attempt `learning_interrupted`. The next launch resumes from the latest checkpoint, on the same host.
- **Test.** Test items run 4 at a time per cell. An interrupted test phase re-runs only the items without a scored result.
- **Automatic resume.** The scheduler re-queues interrupted cells by itself. The back-off doubles with each resume that made no progress (1, 2, 4, 8, 16, then 30 minutes). After 5 such resumes in a row (2 for `crashed`) or 20 in all, the cell waits for a person.
- **Cells that need a person** never start automatically. `phase16.sh status` lists them on its `attention:` line. These are `refused` and `violation` cells, learning stopped by a harness failure, and resume caps reached.
- **Pauses.** No cell is launched while the gateway is paused:
  - a low balance, the key's daily limit, an upstream 402 or an unknown budget is waited out;
  - a spent budget makes the scheduler stop once nothing runs (supervisor exit 10).

### 8.3 Concurrency knobs

| Knob | Set with | Default | Effect |
|---|---|---|---|
| Cells per replicate | `P16_CONCURRENCY` | 4 | `--concurrency` of each scheduler (1–24) |
| Test items per cell | fixed in `phase16.sh` | 4 | `--test-concurrency` |
| WebShop cells | `-- --benchmark-concurrency webshop=N`, `-- --host-benchmark-concurrency webshop=N` | 4 per scheduler, 6 per host | One server, with one global lock, serves every replicate |
| Memory and disk gates | `P16_GATES` | see §5.4 | Launch gate and memory emergency |
| Upstream calls per gateway | fixed (`max_concurrency` in `infra/gateway.py`) | 32 | Calls to OpenRouter in flight at once |
| Postgres connections | `infra/compose.yaml` | 2000 | Each MIRIX server pools up to 80 + 30 connections |

Options after `--` go to the supervisors of that launch only, for example `phase16.sh up r1 -- --host-benchmark-concurrency webshop=4`. Repeat them on every later `up` or `recover`.

### 8.4 Optional steps

Each optional step has its own budget, so it cannot run at the same time as r1–r3 (unless `P16_ALLOW_OVERLAP=1`):

- `phase16.sh up pilot` runs the `phase16-pilot` study: the no-memory method on WebShop's 50-item pilot split, for both models.
- `phase16.sh up dev` runs the `phase16-dev` study: all 16 cells, learning on the whole training split and testing on the whole validation split, never the test split.

### 8.5 Estimating cost

The repository holds no cost figures. Estimate the cost from your own smoke cells:

1. **Run smoke cells** (§7) for each benchmark × model × method you want to price. For MIRIX, use at least as many training items as MIRIX's trigger threshold (`SKILL_TRIGGER_SESSION_THRESHOLD` in `mirix/constants.py`, 5). With fewer, MIRIX learns only in the final flush, never in a regular learning run. The `dev` step (§8.4) gives a closer estimate, on the whole training and validation splits.
2. **Read the cost section** of `report.py --split-mode smoke` (`## Cost` in `_report/report.md`). It gives each cell's cost by role and phase, with the learning side (setup and learning) apart from the test phase.
3. **Compute per-item costs:** the learning cost ÷ N training items, and the test cost ÷ N test items.
4. **Extrapolate** with the split sizes in `SPLIT_SIZES`. ALFWorld has 39 training and 134 test items; WebShop has 40 and 140. `none` cells have no learning side. Sum the 16 cells, then multiply by 3 replicates.
5. **Add a margin.** Smoke cells test on validation items. MIRIX's learning cost per item grows with its skill bank. WebShop requests grow within an episode, because the actor sees the whole episode. The budget is also enforced only from OpenRouter's usage reading, so calls in flight can pass it slightly.

Set `P16_BUDGET` to the result. The key must hold this budget plus the 10 USD pause floor; the preflight `balance` check verifies it.

## 9. Monitoring and recovery

```bash
evals/skill_bench/tools/phase16.sh status            # every step
evals/skill_bench/tools/phase16.sh status r1 r2 r3
```

`status` runs `supervise.py --status` for each step. It shows:
- whether the supervisor and scheduler are alive, and the heartbeat's age;
- cell counts and the next automatic resumes;
- cells that need a person;
- the pause reason, the balance and the budget spent;
- the host's memory, swap, disk and load;
- the current alert.

It exits 1 when a heartbeat is missing or stale, or when the scheduler seems hung. Like `up`, it reads the launcher config first.

Files in each runs directory:

| File | Content |
|---|---|
| `_supervise/heartbeat.json` | Written every 30 s: scheduler counts, pause, budget, host resources (`resources.load` holds the 1-, 5- and 15-minute load averages, so `load[0]` is the 1-minute one), open alerts |
| `_supervise/events.jsonl` | Supervisor events: starts, exits, restarts, alerts, notices |
| `_supervise/ALERT` | The current alert text. The newest alert that needs a person is on the first line. |
| `_supervise/scheduler.log` | The scheduler's log |
| `_schedule/state.json`, `_schedule/events.jsonl` | Scheduler queue, workers and pause, and its event log |
| `_schedule/resources.jsonl` | Host resource samples |
| `<cell_id>/status.json` | Each cell's status |
| `<cell_id>/attempt-NN/failures.jsonl` | Every failure of an attempt, with its kind and recovery |

**Supervisor exit codes** (`tools/supervise.py`):

| Exit | Meaning | What to do |
|---|---|---|
| 0 | Every cell is complete | Go on to §10 |
| 2 | Some cells need a person, or the start was refused | Read `_supervise/ALERT` and the `attention:` line of `status` |
| 10 | The budget is spent | Raise `P16_BUDGET` in the config file (the starting point stays the same), then `phase16.sh up r1 r2 r3` |
| 11 | Invalid budget: the key's usage is below the budget's start, so it is another key | Use the key the budget started on |
| 13 | Another supervisor holds this runs directory | Use the running one |
| 70 | Internal error of the supervisor | Restart it. The systemd unit restarts it by itself. |
| 130 | Stopped by a signal; its workers were drained | Start it again when you are ready |

**After a reboot:**

```bash
cd <checkout>
evals/skill_bench/tools/phase16.sh recover r1 r2 r3
```

`recover` does three things:
1. starts the compose Postgres (`docker compose … up -d`);
2. starts the WebShop server and waits up to `P16_WEBSHOP_WAIT_S` for it;
3. runs `up` for the named steps.

Interrupted learning phases resume from their latest checkpoint, and interrupted test phases re-run their pending items. With the systemd unit, the supervisors restart by themselves. Postgres and the WebShop container come back with the Docker daemon.

**Stopping a supervisor:** press Ctrl-C in its tmux pane, or run `tmux send-keys -t p16-r1 C-c`. The supervisor stops the scheduler, sends `SIGTERM` to its workers, waits for them, and exits 130. Every cell stays resumable.

**Releasing a cell that needs a person.** After you have looked at the cell's `status.json` and `failures.jsonl`, stop that replicate's supervisor and relaunch it with a one-off scheduler option after `--`:

```bash
# Learning stopped by a harness failure: start a new attempt from scratch
evals/skill_bench/tools/phase16.sh up r1 -- --rerun-harness-failures alfworld-luna-awm
# A cell held back by its resume caps: clear its counters
evals/skill_bench/tools/phase16.sh up r1 -- --reset-resume webshop-luna56-mirix
```

The supervisor records each one-off option in `_supervise/one_off.json` and applies it only once.

## 10. Producing results

Everything in this section runs offline: it reads the runs directories and calls no model. Run the audit and the report for each replicate, then the aggregation across replicates:

```bash
cd <checkout>
for r in r1 r2 r3; do
  ALFWORLD_DATA="$HOME/.cache/alfworld" .venv/bin/python -m evals.skill_bench.audit \
    --runs-dir "evals/skill_bench/runs/phase16-$r"
  .venv/bin/python -m evals.skill_bench.report \
    --runs-dir "evals/skill_bench/runs/phase16-$r"
done

.venv/bin/python -m evals.skill_bench.tools.aggregate_replicates --study phase16 \
  evals/skill_bench/runs/phase16-r1 \
  evals/skill_bench/runs/phase16-r2 \
  evals/skill_bench/runs/phase16-r3
```

Run the audit first, because the report and the aggregation read `_audit/audit.json` to mark each cell's audit status. With `--study phase16`, the aggregation selects the study's cells, expects three replicates per cell (`--expect` changes this) and lists any cell of the study that no runs directory holds. The audit needs the ALFWorld game files; WebShop cells are audited from the committed files, without the server.

| Tool | Writes | Exit status |
|---|---|---|
| `evals.skill_bench.audit` | `<runs>/_audit/audit.json`, `audit.md`: per cell, the result of each check and its findings (ARCHITECTURE.md §4.10) | 0 no violation and all checked; 1 a violation; 2 incomplete cells, checks that could not run, or nothing to audit |
| `evals.skill_bench.report` | `<runs>/_report/report.json` (the source of truth), `report.md`. The Markdown holds the cells table, main results per benchmark, model and method (success rate; for WebShop also mean score and success within 15 steps), the comparison with the no-memory cell (an exact McNemar test for success, a 95% paired bootstrap interval for WebShop's mean score), learning curves, cost tables and diagnostics. | 0 written; 2 no complete cell; 3 only with `--require-audit`, when a complete cell has no passing audit |
| `evals.skill_bench.tools.aggregate_replicates` | `evals/skill_bench/runs/_replicates/replicates.json` (the source of truth) and `replicates.md`, in `_replicates/` next to the first runs directory (`--out` to change). The Markdown opens with a **Summary**: per benchmark, each model and method's metrics as mean ± std (n). It then has a **Cells** table (per cell id and metric: each replicate's value, n, mean, sample standard deviation and flags), a **Flags** list and each runs directory's **Spend**. The JSON also holds each metric's min and max. | 0 written; 2 no runs directory holds a complete cell |

Each cell directory also holds its own records: `results.jsonl` (one scored row per test item), `learning.jsonl` (the learning curve), `summary.json`, and per attempt `costs.json` and the trajectories. The architecture document describes the layout (§4.8).

## 11. Troubleshooting

**The WebShop server is not ready.**
- Run `.venv/bin/python -m evals.skill_bench.envs.webshop --status`. It shows one of these states:
  - `loading`: wait. After a start or a crash, the server reloads the product set and the index.
  - `failed`: read its `problems`. A server that runs an image built from other server files, or that differs from the committed split files, fails here. Rebuild with `--build` from this checkout, then `--stop` and `--start --wait-s 1800`.
  - unreachable: the container is not running (`docker ps -a --filter name=skill-bench-webshop`). Start it with `--start`.
- Meanwhile, WebShop cells wait in the scheduler's gate and the other cells go on.

**Preflight reports `key` or `balance` as FAIL.**
- `key`: `--env-file` must name a file with the line `OPENROUTER_API_KEY=…`.
- `balance`: the key's remaining balance, or its limit, must cover 10 USD plus `--budget`. Top up, or raise the key's limit.

**The run pauses.** Run `phase16.sh status` and read the `pause:` line:
- `balance_below_floor`, `upstream_402` or `budget_unknown` lift by themselves, once there is money or a usage reading again;
- `budget_spent` ends the supervisor with exit 10 (§9);
- `budget_invalid` (exit 11) means the key in use is not the one the budget started on.

**A model is unavailable on OpenRouter.**
- `.venv/bin/python -m evals.skill_bench.tools.fetch_models --check` compares the committed `config/models.json` with OpenRouter's public metadata, without a key and without writing anything. It exits 1 on a difference.
- While a model has no available endpoint, its calls are retried as infrastructure failures, and the cells resume later.
- If OpenRouter withdraws `openai/gpt-6-luna` or `openai/gpt-5.6-luna` for good, the study cannot be reproduced with the same models; the code provides no substitute.

**Postgres port or subnet conflicts.**
- `docker compose … up` fails with "port is already allocated": set `SKILL_BENCH_PG_PORT` to a free port, in your shell **and** in the launcher config, then run `up -d` again.
- "Pool overlaps with other one on this space", or no free address pool: set `SKILL_BENCH_SUBNET` to a free private `/24`.
- MIRIX cells refuse to start while Postgres or its template database is missing. Check with `docker compose -f evals/skill_bench/infra/compose.yaml ps`.

**MIRIX cells refuse to start because of a `.env` file.** The message names the file. Move it out of the checkout and every parent directory (§5.3).

**Cells are refused because the gateway estimates tokens from characters.** The `o200k_base` vocabulary is not in the cache the gateway reads. Cache it (§3), make sure `TIKTOKEN_CACHE_DIR` is in the launcher config, and restart the supervisors.

**A learning resume is refused on another machine.** Learning resumes only on the host that took the checkpoint, because the check compares machine ids. Finish each replicate on the host it started on.

## 12. Running the tests

Tests live in `tests/` (MIRIX) and `evals/skill_bench/tests/` (the harness). `pytest.ini` runs both by default and deselects integration tests (`-m "not integration"`). The harness's `conftest.py` removes provider keys while harness tests run, and all model traffic goes to a fake upstream (`evals/skill_bench/tests/fakes.py`).

```bash
# Harness unit tests (no network, no key; needs the cached o200k_base vocabulary)
env -u OPENROUTER_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  .venv/bin/python -m pytest -m "not integration" evals/skill_bench/tests

# MIRIX's unit tests and the harness's in one process
env -u OPENROUTER_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  .venv/bin/python -m pytest -m "not integration" tests evals/skill_bench/tests

# Integration tests: real Postgres and real MIRIX servers, no model calls.
# They create and drop databases in the compose Postgres (SKILL_BENCH_PG_PORT), so run them while no cell runs.
docker compose -f evals/skill_bench/infra/compose.yaml up -d
env -u OPENROUTER_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  .venv/bin/python -m pytest -m integration evals/skill_bench/tests
```

Some tests skip when their inputs are missing:
- tests on real ALFWorld games need `$ALFWORLD_DATA`;
- the byte-level comparison tests need `SKILLOPT_DIR`, `VERL_AGENT_DIR` and `WEBSHOP_UPSTREAM_DIR` (§5.4);
- every harness test skips, with the caching command, while the tokenizer vocabulary is not cached.

`evals/skill_bench/tests/test_phase16_launcher.py` runs `tools/phase16.sh` and the unit template under bash with a fake `tmux`. It needs no tmux, Docker or key.
