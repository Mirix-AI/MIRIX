#!/bin/bash
# What is the graph worth? Four QA-only arms over ONE store — no ingestion anywhere.
#
# Every prior answer is confounded. "+96 questions" came from contaminated prompts with
# each arm on its own ingest. Today's "+9 on 411" used clean prompts but still built two
# separate stores, and re-ingesting identical code moves the score by +5/411 on its own.
#
# All four arms read mirix_locomo_clean_r1 — byte-identical Postgres rows, chunk markers
# seeded from clean_v723_full so nothing is re-ingested. Arm 1 vs arm 2 differ only in
# whether retrieval enters through Neo4j. Arms 3 and 4 repeat 1 and 2 unchanged; their
# spread is the QA-side noise floor, so the effect gets an error bar instead of being read
# off a single pair.
#
# v7.23 is used for the graph arms because it outscores v7.24 on this store (1367 vs 1362,
# multi-hop .883 vs .830) — the graph should be represented by its better retrieval version.
set -u
EVAL=/home/lj/MIRIX_eval; EVD=/home/lj/code/MIRIX/evals
SC=/tmp/claude-1004/-home-lj-code/295438cf-9b9a-4ad5-b513-a0c3840ea4f8/scratchpad
PY=$EVAL/.venv/bin/python; LOG=$SC/ablation.log
DB=mirix_locomo_clean_r1
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
set -a; . $EVAL/.env; set +a
export MIRIX_EVAL_USER_PREFIX=clean-r1__ MIRIX_PG_DB=$DB
echo "== graph ablation, 4 QA-only arms on $DB  $(date '+%F %T') ==" > "$LOG"

run_arm () {
  local name="$1" graph="$2"
  say "---- $name (graph=$graph) ----"
  CURPID=$(ss -ltnp 2>/dev/null | grep ':8531 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "${CURPID:-}" ] && kill "$CURPID" 2>/dev/null
  for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ':8531 ' || break; sleep 1; done
  sed -i "s/^MIRIX_PG_DB=.*/MIRIX_PG_DB=$DB/; \
          s/^MIRIX_GRAPH_VERSION=.*/MIRIX_GRAPH_VERSION=v7.23/; \
          s/^MIRIX_ENABLE_GRAPH_MEMORY=.*/MIRIX_ENABLE_GRAPH_MEMORY=$graph/; \
          s/^MIRIX_GRAPH_RERANK=.*/MIRIX_GRAPH_RERANK=1/" $EVAL/.env
  cd $EVAL && nohup ./run_server.sh > "$SC/server_$name.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s --max-time 3 http://127.0.0.1:8531/health >/dev/null 2>&1 && break; sleep 2
  done
  # Assert. Twice today a run answered with no server and no memory, returned rc=0, and
  # produced a plausible score. A plausible score is not evidence that an arm ran.
  ss -ltn 2>/dev/null | grep -q ':8531 ' || { say "$name SERVER FAILED — arm skipped"; return 1; }

  cd $EVD
  rm -rf "results/locomo/$name" 2>/dev/null
  $PY - "$name" <<'PYEOF'
import json, pathlib, sys
src = pathlib.Path("results/locomo/clean_v723_full")
dst = pathlib.Path("results/locomo") / sys.argv[1]; dst.mkdir(parents=True, exist_ok=True)
n = 0
for f in sorted(src.glob("conv-*.json")):
    if f.name.endswith("_memories.json"):
        continue
    d = json.loads(f.read_text())
    d["records"] = {}                     # re-answer everything; keep the chunk markers
    (dst / f.name).write_text(json.dumps(d, ensure_ascii=False, indent=2))
    n += len(d.get("responses", {}))
print(f"seeded {n} chunk markers")
PYEOF
  $PY -u main_eval.py --data data/locomo10.json --run-llm \
      --mirix_config_path ./configs/0201c_v6.yaml --output_path "$name" \
      > "$SC/${name}_eval.log" 2>&1
  say "$name eval rc=$?"
  $PY -u organize_results.py --judge default "results/locomo/$name" > "$SC/${name}_judge.log" 2>&1
  ACC=$($PY -c "
import json
m=json.load(open('results/locomo/$name/metrics.json'))['overall_metrics']
print(f\"{m['accuracy']:.4f} ({int(m['total_correct'])}/{int(m['total_judged'])})\")" 2>/dev/null)
  # Did this arm use the graph it claims to? A graph arm that silently fell back to flat
  # search, or a no-graph arm that did not, is a different experiment wearing the label.
  G=$(grep -c "graph-owned pass" "$SC/server_$name.log" 2>/dev/null || echo 0)
  say "$name acc=$ACC   graph-owned passes=$G (want >0 iff graph=$graph)"
}

run_arm ab_nograph_r1 false
run_arm ab_graph_r1   true
run_arm ab_nograph_r2 false
run_arm ab_graph_r2   true
say "== all arms done =="
