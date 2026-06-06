#!/bin/bash
# Run MAB Long_Range_Understanding (LRU) eval end-to-end.
#
# Dataset:
#   - infbench_sum_eng_shots2 (100, novel summarisation) ← default
#   - detective_qa            (10, multiple-choice mystery)
# Switch via SOURCE env var (see Optional overrides below).
#
# Judge (auto-routed by source):
#   - infbench_sum_eng_shots2 → mab_summary (gpt-4o-2024-05-13,
#     fluency × precision × recall → F1; aligned with official MAB
#     llm_based_eval/summarization_evaluate.py)
#   - detective_qa            → substring (gold answers are short
#     option strings like "C. The Brandt couple")
# Override with JUDGE env var.
#
# Prereqs:
#   1. `.env` has OPENAI_API_KEY.
#   2. Server is up:   http://localhost:8531/health returns 200.
#      (start with `python scripts/start_server.py --port 8531`)
#   3. `datasets` is installed in the venv.
#
# Optional env overrides:
#   SOURCE=detective_qa    — switch from infbench summarisation
#   LIMIT=N                — number of samples (default: 1)
#   MAX_CHUNKS=N           — cap ingest chunks per sample (default: full)
#   MAX_QS=N               — cap questions per sample (default: all)
#   JUDGE=substring        — override the source-based judge default
#
# Outputs:
#   evals/results/lru/lru_mab_<timestamp>/<sample_id>.json
#   evals/results/lru/lru_mab_<timestamp>/<sample_id>_memories.json
#   evals/results/lru/lru_mab_<timestamp>/metrics.json
#   evals/snapshots/lru_mab_<timestamp>/pg_memory.dump
#   evals/snapshots/lru_mab_<timestamp>/results/

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
OUT="lru_mab_${TS}"

SOURCE_VAL="${SOURCE:-infbench_sum_eng_shots2}"
LIMIT_VAL="${LIMIT:-1}"

# Source -> default judge mapping (overridable via JUDGE).
case "$SOURCE_VAL" in
  infbench_sum_eng_shots2) DEFAULT_JUDGE="mab_summary" ;;
  detective_qa)            DEFAULT_JUDGE="substring"   ;;
  *)                       DEFAULT_JUDGE="mab_summary" ;;
esac
JUDGE_VAL="${JUDGE:-$DEFAULT_JUDGE}"

EXTRA_ARGS=()
[ -n "${MAX_CHUNKS:-}" ] && EXTRA_ARGS+=( "--max-chunks" "$MAX_CHUNKS" )
[ -n "${MAX_QS:-}" ]     && EXTRA_ARGS+=( "--max-questions" "$MAX_QS" )

echo "[1/3] running lru_eval.py -> ${OUT}/"
echo "      source=${SOURCE_VAL}  limit=${LIMIT_VAL}  judge=${JUDGE_VAL}  max_chunks=${MAX_CHUNKS:-<full>}  max_questions=${MAX_QS:-<all>}"
python mab/lru_eval.py \
  --limit "$LIMIT_VAL" \
  --source "$SOURCE_VAL" \
  --run-llm \
  --mirix_config_path ./configs/mab.yaml \
  --output_path "$OUT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "[2/3] running organize_results.py (${JUDGE_VAL} judge)"
python organize_results.py --judge "$JUDGE_VAL" "results/lru/${OUT}"

echo
echo "[3/3] saving snapshot (pg_dump + neo4j export + results copy)"
if MIRIX_PG_DB="${MIRIX_PG_DB:-mirix}" python memory_snapshot.py save "${OUT}" --agents; then
  cp -R "results/lru/${OUT}" "snapshots/${OUT}/results"
  echo "  snapshot + results -> evals/snapshots/${OUT}/"
else
  echo "  WARN: memory_snapshot.py failed; results still at evals/results/lru/${OUT}/" >&2
fi

# ---- summary ---------------------------------------------------------------
echo
echo "========================================================"
echo "  Summary"
echo "========================================================"
python - "results/lru/${OUT}" <<'PY'
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
m = json.loads((out_dir / "metrics.json").read_text())
mt = m["metrics"]

print(f"  Output:           {out_dir}")
acc = mt.get('accuracy')
if acc is not None:
    print(f"  Acc / mean F1:    {acc:.4f}  ({mt['total_correct']:.2f}/{mt['total_judged']})")
print(f"  Total questions:  {mt.get('total_questions')}")
lat = mt.get('average_answer_latency_seconds')
if lat is not None:
    print(f"  Avg answer lat.:  {lat:.2f}s")
print(f"  Avg mem tokens:   {mt.get('average_memory_tokens')}")

# Per-sub-score breakdown when mab_summary was used.
for k in ("gpt-4-fluency", "gpt-4-recall", "gpt-4-precision", "gpt-4-f1"):
    v = mt.get(k)
    if v is not None:
        print(f"  {k:20s} {v:.4f}")
PY

# ---- DB memory counts ------------------------------------------------------
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
