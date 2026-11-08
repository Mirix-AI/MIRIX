#!/bin/bash
# SHDocQA on v7.1, twice: 4096 TOKENS against 4096 CHARACTERS.
#
# The repo records the char-based budget as a bug — "4096 chars is about 1024 tokens, so it
# emitted ~4x more chunks than official, which made MIRIX's retrieval look worse than
# apples-to-apples because each semantic unit was scattered across multiple memories." That
# claim has never been measured; the policy was changed and the old numbers were abandoned.
# This measures it, on the branch where the historical SHDocQA numbers came from.
#
# Two arms, each with its own Postgres database AND its own graph namespace. Sharing either
# is how a run in this project once scored 0.882 against 959 anchors it had not created.
set -u
EVAL=/home/lj/MIRIX_eval
V71=/tmp/v71
SC=/tmp/claude-1004/-home-lj-code/295438cf-9b9a-4ad5-b513-a0c3840ea4f8/scratchpad
PY=$EVAL/.venv/bin/python
PGB=$EVAL/pgenv/bin
LOG=$SC/v71_chunk_ab.log
SOURCE=ruler_qa1_197K            # SHDocQA — single-hop, SQuAD-derived
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
set -a; . $EVAL/.env; set +a
export PGPASSWORD="${MIRIX_PG_PASSWORD:-mirix}"
unset MIRIX_COLDFACT MIRIX_HYBRID_SEARCH MIRIX_SEARCH_LIMIT 2>/dev/null || true
echo "== v7.1 SHDocQA: token vs char chunking  $(date '+%F %T') ==" > "$LOG"

run_arm () {
  local unit="$1" db="$2" prefix="$3" out="$4"
  say "---- $out (MIRIX_CHUNK_UNIT=$unit) ----"

  $PGB/psql -w -h localhost -U mirix -d postgres -q \
    -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$db';" >/dev/null 2>&1
  $PGB/dropdb -w -h localhost -U mirix --if-exists "$db" >/dev/null 2>&1
  $PGB/createdb -w -h localhost -U mirix -T template0 "$db" || { say "createdb FAILED"; return 1; }
  $PGB/psql -w -h localhost -U mirix -d "$db" -q -c "CREATE EXTENSION IF NOT EXISTS vector;"

  EVAL_PREFIX="$prefix" $PY - >>"$LOG" 2>&1 <<'PYEOF'
import os
from neo4j import GraphDatabase
p = os.environ["EVAL_PREFIX"]
d = GraphDatabase.driver(os.environ["MIRIX_NEO4J_URI"],
                         auth=(os.environ["MIRIX_NEO4J_USER"],
                               os.environ["MIRIX_NEO4J_PASSWORD"]))
with d.session() as s:
    n = s.run("MATCH (x) WHERE x.user_id STARTS WITH $p DETACH DELETE x RETURN count(*) AS c",
              p=p).single()["c"]
print(f"wiped {n} nodes under {p!r}")
d.close()
PYEOF

  CURPID=$(ss -ltnp 2>/dev/null | grep ':8531 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "${CURPID:-}" ] && kill "$CURPID" 2>/dev/null
  for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ':8531 ' || break; sleep 1; done

  cd $V71
  MIRIX_PG_DB="$db" MIRIX_GRAPH_VERSION=v7.1 MIRIX_ENABLE_GRAPH_MEMORY=true \
  MIRIX_CHUNK_UNIT="$unit" PGPASSWORD="$PGPASSWORD" \
  nohup $PY scripts/start_server.py --port 8531 > "$SC/server_$out.log" 2>&1 &
  for i in $(seq 1 60); do
    curl -s --max-time 3 http://127.0.0.1:8531/health >/dev/null 2>&1 && break; sleep 2
  done
  # Assert. A run that answers with no server still exits 0 and produces a plausible score.
  ss -ltn 2>/dev/null | grep -q ':8531 ' || { say "$out SERVER FAILED — arm skipped"; return 1; }
  say "$out server up (db=$db, unit=$unit)"

  cd $V71/evals
  say "ENV CHECK prefix=$prefix unit=$unit db=$db"
  rm -rf "results/ruler/$out" 2>/dev/null
  # NOTE: no --max-chunks. In this runner `chunks = chunks[: args.max_chunks]`, so the
  # historical scripts' `--max-chunks 0` means "keep zero chunks", not "no cap" — it deletes
  # the ingest entirely and the run then answers every question against whatever the store
  # already held, exiting 0 either way.
  #
  # ruler_eval.py, NOT longmem_eval.py. Both load the dataset, but longmem_eval parses the
  # context with parse_sessions, which expects LongMemEval's "Chat Time" structure — on a
  # RULER document that yields ZERO chunks, and the run then answers 100 questions against
  # an empty store while exiting 0. ruler_eval has parse_documents for exactly this, and
  # its default source on this branch is already ruler_qa1_197K, which is SHDocQA.
  MIRIX_PG_DB="$db" MIRIX_GRAPH_VERSION=v7.1 MIRIX_ENABLE_GRAPH_MEMORY=true \
  MIRIX_EVAL_USER_PREFIX="$prefix" MIRIX_CHUNK_UNIT="$unit" \
  $PY -u mab/ruler_eval.py --limit 1 --run-llm \
      --output_path "$out" \
      --mirix_config_path ./configs/mab.yaml > "$SC/${out}_eval.log" 2>&1
  say "$out eval rc=$?"

  $PY -u organize_results.py --mab-judge "results/ruler/$out" > "$SC/${out}_judge.log" 2>&1

  # Verification column: chunk count is the whole independent variable, so read it back.
  $PY - "$out" "$db" "$prefix" <<'PYEOF' | tee -a "$LOG"
import json, glob, os, subprocess, sys
out, db, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
d = f"results/ruler/{out}"
fs = [f for f in glob.glob(d + "/*.json") if "metrics" not in f and "memories" not in f]
chunks = sum(len(json.load(open(f)).get("responses", {})) for f in fs)
qs = sum(len(json.load(open(f)).get("records", {})) for f in fs)
ing = sum(sum(float(v) for v in (json.load(open(f)).get("timings", {}).get("add_chunk") or {}).values())
          for f in fs)
acc = ""
p = d + "/metrics.json"
if os.path.exists(p):
    J = json.load(open(p)).get("llm_judge_results") or []
    if J:
        c = sum(1 for x in J if x.get("score") == 1)
        acc = f"{c}/{len(J)} = {c/len(J):.4f}"
rows = subprocess.run(["/home/lj/MIRIX_eval/pgenv/bin/psql", "-w", "-h", "localhost", "-U",
                       "mirix", "-d", db, "-At", "-c",
                       f"SELECT count(*) FROM episodic_memory WHERE NOT is_deleted "
                       f"AND user_id LIKE '{prefix}%';"],
                      capture_output=True, text=True,
                      env={**os.environ, "PGPASSWORD": "mirix"}).stdout.strip()
stray = subprocess.run(["/home/lj/MIRIX_eval/pgenv/bin/psql", "-w", "-h", "localhost", "-U",
                        "mirix", "-d", db, "-At", "-c",
                        f"SELECT count(*) FROM episodic_memory WHERE NOT is_deleted "
                        f"AND user_id NOT LIKE '{prefix}%';"],
                       capture_output=True, text=True,
                       env={**os.environ, "PGPASSWORD": "mirix"}).stdout.strip()
flag = "   <<< ZERO CHUNKS: nothing ingested, score meaningless" if chunks == 0 else ""
print(f"  {out}: acc={acc}  chunks={chunks}  questions={qs}  "
      f"ingest={ing/60:.1f}min  episodic={rows}  outside-prefix={stray}{flag}")
PYEOF
}

run_arm token mirix_v71shdoc_tok v71tok__ v71_shdoc_token
run_arm char  mirix_v71shdoc_chr v71chr__ v71_shdoc_char
say "== both arms done =="
