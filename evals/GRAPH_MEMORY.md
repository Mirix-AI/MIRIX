# Graph memory: running the evaluated configurations

This page covers the settings and commands for MIRIX with the provenance hypergraph. The
flat-memory instructions in [README.md](README.md) still apply; everything below is in
addition to them.

## Requirements

- PostgreSQL with `pgvector`, as for flat MIRIX.
- Neo4j 5.23 or later. The graph queries use `CALL (a, row) { ... }`, which older servers
  reject. `docker compose --profile graph up -d` starts Neo4j 5.26 with APOC.
- The Python `neo4j` driver. It is in `requirements.txt`; after `pip install -e .` also run
  `pip install "neo4j>=5.20,<6"`.
- A database created before graph memory existed needs
  `scripts/migrate_add_graph_provenance_columns.sql` once. The server stops at startup and
  names the script if it is missing.

## Server settings

| variable | effect |
|---|---|
| `MIRIX_ENABLE_GRAPH_MEMORY` | `true` indexes records into the hypergraph and retrieves through it; `false` retrieves from flat Postgres |
| `MIRIX_GRAPH_VERSION` | selects the graph writer and reader policies; the default `v5` keeps the original MIRIX behaviour |
| `MIRIX_GRAPH_RERANK` | `1` reranks graph candidates; `0` turns reranking off for ablations |
| `MIRIX_NEO4J_URI`, `MIRIX_NEO4J_USER`, `MIRIX_NEO4J_PASSWORD` | Neo4j connection (default `bolt://localhost:7687`) |
| `MIRIX_NEO4J_VECTOR_DIM` | dimension of the anchor-name index, 1536 by default; it must match the embedding model |

The method described in the paper corresponds to `MIRIX_GRAPH_VERSION=v7.24`. The LoCoMo
stores behind the reported results were written at `v7.23`.

Start the server from the repository root with these variables set:

```bash
python scripts/start_server.py --port 8531
```

The evaluation clients connect to `http://127.0.0.1:8531`.

## The three configurations

| configuration | `MIRIX_ENABLE_GRAPH_MEMORY` (server) | `MIRIX_GRAPH_VERSION` (server) | `MIRIX_DREAM_EVERY_N_CHUNKS` (eval) |
|---|---|---|---|
| M0, flat memory without AutoDream | `false` | `v7.24` | `0` |
| M1, graph without AutoDream | `true` | `v7.24` | `0` |
| M2, graph with AutoDream | `true` | `v7.24` | `5` |

M0 keeps `MIRIX_GRAPH_VERSION` set, so the Postgres side of the write path (source
metadata, provenance columns, merge rules) matches M1, while nothing is written to or read
from Neo4j. `MIRIX_DREAM_EVERY_N_CHUNKS` is read by the evaluation script, which triggers one
AutoDream cycle every N ingested chunks and after the last one (`MIRIX_DREAM_MODE` defaults
to `experience`).

## LoCoMo

Put `locomo10.json` under `evals/data/`, then from `evals/`:

```bash
unset MIRIX_COLDFACT MIRIX_HYBRID_SEARCH MIRIX_SEARCH_LIMIT   # experiment flags; leave unset
export MIRIX_EVAL_USER_PREFIX=m2__                            # namespaces this run's users
export MIRIX_DREAM_EVERY_N_CHUNKS=5                           # M2; use 0 for M0 and M1
python main_eval.py --data data/locomo10.json --run-llm \
    --mirix_config_path ./configs/0201c_v6.yaml --output_path m2
python organize_results.py --judge default results/locomo/m2
```

Give every run its own `MIRIX_EVAL_USER_PREFIX`; the Neo4j graph is shared across runs.

**Answering again without re-ingesting.** A change that does not alter what is written can be
measured on an existing store. Seed a new output directory from a finished run with its
answers cleared; `main_eval.py` then sees the chunk markers and only answers:

```python
import json, pathlib
src = pathlib.Path("results/locomo/m2")
dst = pathlib.Path("results/locomo/m2_qa_only"); dst.mkdir(parents=True, exist_ok=True)
for f in src.glob("conv-*.json"):
    if f.name.endswith("_memories.json"):
        continue
    d = json.loads(f.read_text())
    d["records"] = {}
    (dst / f.name).write_text(json.dumps(d, ensure_ascii=False, indent=2))
```

**Before scoring a run**, check that it wrote where it should:

```bash
python assert_store_sane.py --db <postgres-db> --prefix m2__ --graph   # PSQL=/path/to/psql if not on PATH
```

## LongMemEval-S and MemoryAgentBench

`longmem_eval.py` runs LongMemEval-S with the same scaffolding as `main_eval.py`. The scripts
in `mab/` run the MemoryAgentBench tracks with `configs/mab.yaml`; each lists its
prerequisites and options at the top. `mab/run_mab_longmem_eval.sh` also snapshots the store
afterwards with `memory_snapshot.py`.
