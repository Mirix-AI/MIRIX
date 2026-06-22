#!/bin/bash
# Run MAB RULER QA eval (SHDOCQA / MHDOCQA) end-to-end.
#
# Dataset:
#   - SHDOCQA = ruler_qa1_197K (single-hop, SQuAD-derived)   ← default
#   - MHDOCQA = ruler_qa2_421K (multi-hop, HotpotQA-derived)
# Switch via SOURCE env var (see Optional overrides below).
#
# Prereqs:
#   1. `.env` has OPENAI_API_KEY.
#   2. Server is up:   http://localhost:8531/health returns 200.
#      (start with `python scripts/start_server.py --port 8531`)
#   3. `datasets` is installed in the venv.
#
# Optional env overrides:
#   SOURCE=ruler_qa2_421K  — switch to MHDOCQA (default is SHDOCQA / qa1_197K)
#   LIMIT=N                — number of conversations (default: 1)
#   MAX_CHUNKS=N           — cap ingest chunks per conv (default: full)
#   MAX_QS=N               — cap QA questions per conv (default: all 100)
#
# Outputs:
#   evals/results/ruler/ruler_mab_<timestamp>/ruler_qa_0.json     — per-question records
#   evals/results/ruler/ruler_mab_<timestamp>/ruler_qa_0_memories.json
#   evals/results/ruler/ruler_mab_<timestamp>/metrics.json        — substring judge accuracy
#   evals/snapshots/ruler_mab_<timestamp>/pg_memory.dump          — full PG dump
#   evals/snapshots/ruler_mab_<timestamp>/results/                — copy of the result dir

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
OUT="ruler_mab_${TS}"

SOURCE_VAL="${SOURCE:-ruler_qa1_197K}"
LIMIT_VAL="${LIMIT:-1}"

EXTRA_ARGS=()
[ -n "${MAX_CHUNKS:-}" ] && EXTRA_ARGS+=( "--max-chunks" "$MAX_CHUNKS" )
[ -n "${MAX_QS:-}" ]     && EXTRA_ARGS+=( "--max-questions" "$MAX_QS" )

echo "[1/3] running ruler_eval.py -> ${OUT}/"
echo "      source=${SOURCE_VAL}  limit=${LIMIT_VAL}  max_chunks=${MAX_CHUNKS:-<full>}  max_questions=${MAX_QS:-<all 100>}"
python mab/ruler_eval.py \
  --limit "$LIMIT_VAL" \
  --source "$SOURCE_VAL" \
  --run-llm \
  --mirix_config_path ./configs/mab.yaml \
  --output_path "$OUT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "[2/3] running organize_results.py (substring judge)"
# RULER's official metric is substring_exact_match — no LLM judge needed.
python organize_results.py --judge substring "results/ruler/${OUT}"

echo
echo "[3/3] saving snapshot (pg_dump + neo4j export + results copy)"
if MIRIX_PG_DB="${MIRIX_PG_DB:-mirix}" python memory_snapshot.py save "${OUT}" --agents; then
  cp -R "results/ruler/${OUT}" "snapshots/${OUT}/results"
  echo "  snapshot + results -> evals/snapshots/${OUT}/"
else
  echo "  WARN: memory_snapshot.py failed; results still at evals/results/ruler/${OUT}/" >&2
fi

# ---- summary ---------------------------------------------------------------
echo
echo "========================================================"
echo "  Summary"
echo "========================================================"
python - "results/ruler/${OUT}" <<'PY'
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

# RULER has no question_types, so accuracy_by_category is uninformative.
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
