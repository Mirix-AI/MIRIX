#!/bin/bash
# Full LoCoMo re-run on DECONTAMINATED prompts (commit 36d8d40).
#
# Sixteen LoCoMo gold answers had been written verbatim into the system prompts on both
# sides of the pipeline — three in the answerer, thirteen in the ingest prompts under
# evals/prompts/0201a/. The ingest side is why this cannot be a QA-only rerun: those
# examples shaped how memories were WRITTEN, so the graph and the PG rows themselves
# carry the contamination and have to be rebuilt.
#
# Replicates the setup that produced the 1371/1540 figure, changing only the prompts:
#   step 1  full ingest + QA at v7.23   (builds the graph)
#   step 2  QA-only at v7.24 on that graph, which is what 1371/1540 was
#
# Roughly 4.5h ingest + ~1.5h QA per step. Everything is saved so the comparison can be
# rerun without paying the ingest again.
set -u
EVAL=/home/lj/MIRIX_eval
SC=/tmp/claude-1004/-home-lj-code/295438cf-9b9a-4ad5-b513-a0c3840ea4f8/scratchpad
PY=$EVAL/.venv/bin/python
PGB=$EVAL/pgenv/bin
EVD=/home/lj/code/MIRIX/evals
DB=mirix_p3_nograph
PREFIX=p3ng__
LOG=$SC/p3nograph.log
BK=$EVAL/saved/p3_nograph
mkdir -p "$BK"

set -a; . $EVAL/.env; set +a
export PGPASSWORD="${MIRIX_PG_PASSWORD:-mirix}"
unset MIRIX_COLDFACT MIRIX_HYBRID_SEARCH MIRIX_SEARCH_LIMIT 2>/dev/null || true
export MIRIX_EVAL_USER_PREFIX="$PREFIX"
# Matches the contaminated run being replaced: dream every 5 chunks.
export MIRIX_DREAM_EVERY_N_CHUNKS=5
export MIRIX_DREAM_MODE=experience

say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
echo "== locomo clean re-run  started $(date '+%F %T') ==" > "$LOG"

# ---- fresh PG ----
$PGB/psql -w -h localhost -U mirix -d postgres -q \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$DB';" >/dev/null 2>&1
$PGB/dropdb -w -h localhost -U mirix --if-exists "$DB" >/dev/null 2>&1
$PGB/createdb -w -h localhost -U mirix -T template0 "$DB" || { say "createdb FAILED"; exit 1; }
$PGB/psql -w -h localhost -U mirix -d "$DB" -q -c "CREATE EXTENSION IF NOT EXISTS vector;"
say "fresh db $DB"

# ---- fresh graph for this prefix only ----
EVAL_PREFIX="$PREFIX" $PY - >>"$LOG" 2>&1 <<'PYEOF'
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ['MIRIX_NEO4J_URI'],
                         auth=(os.environ['MIRIX_NEO4J_USER'], os.environ['MIRIX_NEO4J_PASSWORD']))
p = os.environ["EVAL_PREFIX"]
with d.session() as s:
    r = s.run("MATCH (n) WHERE n.user_id STARTS WITH $p DETACH DELETE n "
              "RETURN count(n) AS c", p=p).single()
    print(f"wiped {r['c']} nodes under {p!r}")
d.close()
PYEOF

start_server () {
  local ver="$1" tag="$2"
  CURPID=$(ss -ltnp 2>/dev/null | grep ':8531 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "${CURPID:-}" ] && kill "$CURPID" 2>/dev/null
  for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ':8531 ' || break; sleep 1; done
  sed -i "s/^MIRIX_PG_DB=.*/MIRIX_PG_DB=$DB/; \
          s/^MIRIX_GRAPH_VERSION=.*/MIRIX_GRAPH_VERSION=$ver/; \
          s/^MIRIX_ENABLE_GRAPH_MEMORY=.*/MIRIX_ENABLE_GRAPH_MEMORY=false/; \
          s/^MIRIX_GRAPH_RERANK=.*/MIRIX_GRAPH_RERANK=0/" $EVAL/.env
  export MIRIX_GRAPH_VERSION="$ver"
  cd $EVAL && nohup ./run_server.sh > "$SC/server_p3ng_$tag.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s --max-time 3 http://127.0.0.1:8531/health >/dev/null 2>&1 && break; sleep 2
  done
  ss -ltn 2>/dev/null | grep -q ':8531 ' || { say "SERVER FAILED ($tag)"; exit 1; }
  say "server up (db=$DB, graph=$ver)"
}

judge_and_report () {
  local out="$1"
  cd $EVD
  $PY -u organize_results.py --judge default "results/locomo/$out" \
      > "$SC/p3ng_${out}_judge.log" 2>&1
  ACC=$($PY -c "
import json
m=json.load(open('results/locomo/$out/metrics.json'))['metrics']
print(f\"{m['accuracy']:.4f} ({int(m['total_correct'])}/{int(m['total_judged'])})\")" \
      2>/dev/null || echo MISSING)
  say "$out acc=$ACC   [clean v7.23 with graph, same 3 convs: 0.8881 (365/411)]"
}

# ---- step 1: ingest + QA at v7.23 ----
start_server v7.23 v723
cd $EVD
rm -rf results/locomo/p3ng_full 2>/dev/null
$PY -u main_eval.py --data data/locomo_pilot3.json --run-llm \
    --mirix_config_path ./configs/0201c_v6.yaml --output_path p3ng_full \
    > "$SC/p3ng_eval.log" 2>&1
say "step 1 eval rc=$?"
judge_and_report p3ng_full

$PGB/pg_dump -w -h localhost -U mirix -d "$DB" -f "$BK/$DB.sql" \
  && say "pg dump -> $BK/$DB.sql ($(du -h "$BK/$DB.sql" | cut -f1))"


# Verification column. Two things must be true for this arm to mean anything: the rows
# belong to THIS run, and the graph really is off. Today a run wrote every row to the bare
# "conv-43" namespace, mixed with an older run's 959 anchors in the shared Neo4j, returned
# rc=0 and a plausible 0.882, and nothing in the output said the store was not its own.
cd $EVAL
R=$(PGPASSWORD=mirix $PGB/psql -w -h localhost -U mirix -d "$DB" -At \
    -c "SELECT count(*) FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'p3ng__%';")
B=$(PGPASSWORD=mirix $PGB/psql -w -h localhost -U mirix -d "$DB" -At \
    -c "SELECT count(*) FROM episodic_memory WHERE NOT is_deleted AND user_id NOT LIKE 'p3ng__%';")
G=$($PY -c "
import os
from neo4j import GraphDatabase
d=GraphDatabase.driver(os.environ['MIRIX_NEO4J_URI'],auth=(os.environ['MIRIX_NEO4J_USER'],os.environ['MIRIX_NEO4J_PASSWORD']))
with d.session() as s:
    print(s.run(\"MATCH (n) WHERE n.user_id STARTS WITH 'p3ng__' RETURN count(*) AS c\").single()['c'])
d.close()" 2>/dev/null | tail -1)
say "VERIFY prefixed rows=$R  unprefixed=$B (want 0)  graph nodes=$G (want 0)"
say "== done (no graph, clean prompts, 3 convs) =="
