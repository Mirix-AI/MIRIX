#!/bin/bash
# Run MAB-enabled eval (conflict resolution + source provenance) on
# LongMemEval-S (HuggingFace ai-hyz/MemoryAgentBench) end-to-end.
#
# Prereqs:
#   1. `.env` has OPENAI_API_KEY.
#   2. Server is up:   http://localhost:8531/health returns 200.
#      (start with `python scripts/start_server.py --port 8531`)
#   3. `datasets` is installed in the venv (the HF library); the runner
#      raises a clear error if not.
#
# Optional env overrides (handy for quick tests):
#   LIMIT=N        — how many LongMemEval-S conversations to ingest (default: 1)
#   MAX_CHUNKS=N   — cap ingest chunks per conversation (default: no cap, full conv)
#   MAX_QS=N       — cap QA questions per conversation (default: no cap, all 60)
#
# Outputs:
#   evals/results/longmem/longmem_mab_<timestamp>/longmem_s_0.json     — per-question records
#   evals/results/longmem/longmem_mab_<timestamp>/longmem_s_0_memories.json
#   evals/results/longmem/longmem_mab_<timestamp>/metrics.json         — LLM-judge accuracy
#   evals/snapshots/longmem_mab_<timestamp>/pg_memory.dump             — full PG dump
#   evals/snapshots/longmem_mab_<timestamp>/results/                   — copy of the result dir above
#   evals/snapshots/longmem_mab_<timestamp>/{meta,neo4j_graph}.json    — snapshot metadata

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVALS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EVALS_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# ---- pre-flight ------------------------------------------------------------
if ! lsof -ti:8531 >/dev/null 2>&1; then
  echo "ERROR: nothing listening on :8531" >&2
  echo "       start the server first: python scripts/start_server.py --port 8531" >&2
  exit 1
fi

python - <<'PY' || { echo "ERROR: server /health not 200" >&2; exit 1; }
import sys, urllib.request
try:
    r = urllib.request.urlopen("http://localhost:8531/health", timeout=5)
    sys.exit(0 if r.status == 200 else 2)
except Exception as e:
    print(f"  health probe: {e}", file=sys.stderr)
    sys.exit(3)
PY

python - <<'PY' || { echo "ERROR: \`datasets\` not installed in this Python (uv pip install datasets)" >&2; exit 1; }
import importlib, sys
sys.exit(0 if importlib.util.find_spec("datasets") else 1)
PY

# ---- run -------------------------------------------------------------------
cd "${EVALS_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="longmem_mab_${TS}"

# Assemble optional caps from env
EXTRA_ARGS=()
[ -n "${MAX_CHUNKS:-}" ]    && EXTRA_ARGS+=( "--max-chunks"    "$MAX_CHUNKS" )
[ -n "${MAX_QS:-}" ]        && EXTRA_ARGS+=( "--max-questions" "$MAX_QS" )
LIMIT_VAL="${LIMIT:-1}"

echo "[1/3] running longmem_eval.py (MAB-enabled) -> ${OUT}/"
echo "      limit=${LIMIT_VAL}  max_chunks=${MAX_CHUNKS:-<full>}  max_questions=${MAX_QS:-<all 60>}"
python mab/longmem_eval.py \
  --limit "$LIMIT_VAL" \
  --run-llm \
  --mirix_config_path ./configs/mab.yaml \
  --output_path "$OUT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "[2/3] running organize_results.py (LLM judge)"
# longmem_eval writes under results/longmem/<OUT>/; organize_results.py
# defaults its fallback path to results/locomo/, so we pass the full path.
python organize_results.py --mab-judge "results/longmem/${OUT}"

echo
echo "[3/3] saving snapshot (pg_dump + neo4j export + results copy)"
# Non-fatal: the eval + judge already succeeded by here. If the snapshot
# tool can't reach pg_dump or the neo4j driver is missing we still want
# the run to be considered successful.
if MIRIX_PG_DB="${MIRIX_PG_DB:-mirix}" python memory_snapshot.py save "${OUT}" --agents; then
  # Co-locate the results inside the snapshot directory so a single
  # archive folder has everything needed to verify later.
  cp -R "results/longmem/${OUT}" "snapshots/${OUT}/results"
  echo "  snapshot + results -> evals/snapshots/${OUT}/"
else
  echo "  WARN: memory_snapshot.py failed; results still at evals/results/longmem/${OUT}/" >&2
fi

# ---- summary ---------------------------------------------------------------
echo
echo "========================================================"
echo "  Summary"
echo "========================================================"
python - "results/longmem/${OUT}" <<'PY'
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
m = json.loads((out_dir / "metrics.json").read_text())
mt = m["metrics"]

print(f"  Output:           {out_dir}")
print(f"  Acc:              {mt['accuracy']:.4f}  ({mt['total_correct']}/{mt['total_judged']})")
print(f"  Total questions:  {mt.get('total_questions')}")
print(f"  Avg answer lat.:  {mt.get('average_answer_latency_seconds', 0):.2f}s")
print(f"  Avg mem tokens:   {mt.get('average_memory_tokens')}")

cat_acc = m.get("accuracy_by_category") or {}
if cat_acc:
    print()
    print("  Per category:")
    for k, v in sorted(cat_acc.items()):
        n = v.get("total_judged", 0)
        a = v.get("accuracy", 0)
        print(f"    {str(k):20s}  n={n:3d}  acc={a:.4f}")
PY

# ---- DB memory counts (handy for cross-run comparison) ---------------------
echo
echo "========================================================"
echo "  Memory store counts (DB=\${MIRIX_PG_DB:-mirix})"
echo "========================================================"
PGPASSWORD="${MIRIX_PG_PASSWORD:-mirix}" psql -h "${MIRIX_PG_HOST:-localhost}" \
  -U "${MIRIX_PG_USER:-mirix}" -d "${MIRIX_PG_DB:-mirix}" -tAc \
  "select 'episodic='||count(*) from episodic_memory
   union all select 'semantic='||count(*) from semantic_memory
   union all select 'procedural='||count(*) from procedural_memory
   union all select 'resource='||count(*) from resource_memory
   union all select 'knowledge_vault='||count(*) from knowledge_vault" \
  2>/dev/null || echo "  (psql unavailable or DB unreachable; skipping)"
